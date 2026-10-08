# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def _load_post_load_method(*, clock, config, convert_nz):
    # Execute the real method without importing its NPU-only module dependencies.
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/quantization/methods/w8a8_dynamic.py"
    tree = ast.parse(source.read_text())
    scheme = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendW8A8DynamicLinearMethod"
    )
    method = next(
        node
        for node in scheme.body
        if isinstance(node, ast.FunctionDef) and node.name == "process_weights_after_loading"
    )
    namespace = {
        "torch": torch,
        "time": SimpleNamespace(perf_counter=clock),
        "get_ascend_config": config,
        "maybe_trans_nz": convert_nz,
        "enable_dsa_cp": Mock(side_effect=AssertionError("wq_a must not query the wq_b workaround")),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["process_weights_after_loading"]


def _layer():
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.self_attn.wq_a"
    layer.weight = torch.nn.Parameter(torch.arange(12, dtype=torch.int8).reshape(3, 4), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        torch.tensor([[0.5], [1.5], [2.5]], dtype=torch.float16), requires_grad=False
    )
    layer.weight_offset = torch.nn.Parameter(
        torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float16), requires_grad=False
    )
    return layer


@pytest.mark.parametrize("diagnostics", [False, True])
def test_w8a8_first_linear_stage_timing_preserves_tensor_results(diagnostics):
    layer = _layer()
    original_weight = layer.weight.detach().clone()
    original_scale = layer.weight_scale.detach().clone()
    original_offset = layer.weight_offset.detach().clone()
    if diagnostics:
        layer._rfork_post_load_stages = {}
        clock = Mock(side_effect=[0, 2, 2, 5, 5, 9])
        config = Mock(return_value=SimpleNamespace(weight_nz_mode=2))
    else:
        clock = Mock(side_effect=AssertionError("diagnostics disabled must not read the clock"))
        config = Mock(side_effect=AssertionError("diagnostics disabled must not query NZ metadata"))
    convert_nz = Mock(side_effect=lambda tensor: tensor.clone())
    process = _load_post_load_method(clock=clock, config=config, convert_nz=convert_nz)

    process(SimpleNamespace(act_quant_type=torch.int8), layer)

    convert_nz.assert_called_once()
    torch.testing.assert_close(layer.weight, original_weight.transpose(0, 1).contiguous())
    assert layer.weight.is_contiguous()
    torch.testing.assert_close(layer.weight_scale, original_scale.flatten())
    torch.testing.assert_close(layer.weight_scale_fp32, original_scale.flatten().to(torch.float32))
    torch.testing.assert_close(layer.weight_offset, original_offset.flatten())
    if diagnostics:
        config.assert_called_once_with()
        assert clock.call_count == 6
        assert layer._rfork_post_load_stages == {
            "weight_shape": (3, 4),
            "weight_dtype": "torch.int8",
            "weight_stride": (4, 1),
            "weight_logical_bytes": 12,
            "scale_shape": (3, 1),
            "scale_dtype": "torch.float16",
            "weight_nz_mode": 2,
            "transpose_contiguous_host_s": 2,
            "nz_host_s": 3,
            "scale_fp32_host_s": 4,
        }
    else:
        clock.assert_not_called()
        config.assert_not_called()
        assert not hasattr(layer, "_rfork_post_load_stages")


def test_w8a8_nz_failure_preserves_completed_stage_timing():
    layer = _layer()
    original_weight = layer.weight.detach().clone()
    original_scale = layer.weight_scale.detach().clone()
    layer._rfork_post_load_stages = {}
    error = RuntimeError("NZ conversion failed")
    convert_nz = Mock(side_effect=error)
    process = _load_post_load_method(
        clock=Mock(side_effect=[0, 2, 2]),
        config=Mock(return_value=SimpleNamespace(weight_nz_mode=2)),
        convert_nz=convert_nz,
    )

    with pytest.raises(RuntimeError) as raised:
        process(SimpleNamespace(act_quant_type=torch.int8), layer)

    assert raised.value is error
    convert_nz.assert_called_once()
    assert layer._rfork_post_load_stages["transpose_contiguous_host_s"] == 2
    assert "nz_host_s" not in layer._rfork_post_load_stages
    assert "scale_fp32_host_s" not in layer._rfork_post_load_stages
    torch.testing.assert_close(layer.weight, original_weight.transpose(0, 1).contiguous())
    torch.testing.assert_close(layer.weight_scale, original_scale)
