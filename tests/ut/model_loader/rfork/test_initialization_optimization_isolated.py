# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


def load_function(path, name, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if class_name:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace[name]


def test_quant_type_uses_current_exact_weight_and_checks_fused_shards():
    namespace = {"MappingProxyType": MappingProxyType, "logger": Mock()}
    load_function("vllm_ascend/quantization/modelslim_config.py", "_is_missing_v_shard", namespace)
    load_function("vllm_ascend/quantization/modelslim_config.py", "get_linear_quant_type", namespace)
    lookup = load_function(
        "vllm_ascend/quantization/modelslim_config.py", "is_layer_skipped_ascend", namespace, "AscendModelSlimConfig"
    )
    description = {"layer.weight": "INT8", "layer.child.weight": "FLOAT"}
    config = SimpleNamespace(quant_description=description)
    assert not lookup(config, "layer")
    description["layer.weight"] = "FLOAT"
    assert lookup(config, "layer")
    del description["layer.weight"]
    assert lookup(config, "layer")  # A missing exact key is unquantized in v0.30.
    description.update({"q.weight": "INT8", "k.weight": "FLOAT"})
    with pytest.raises(ValueError):
        lookup(config, "qkv", {"qkv": ["q", "k"]})


@pytest.mark.parametrize("model_type", ["deepseek_v4", "glm_moe_dsa"])
def test_target_model_expert_mapping_uses_each_quantized_shard(model_type):
    path = "vllm_ascend/quantization/modelslim_config.py"
    tree = ast.parse((ROOT / path).read_text())
    mappings = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "packed_modules_model_mapping"
    )
    entries = dict(zip(mappings.keys, mappings.values, strict=True))
    model_mapping = next(
        value for key, value in entries.items() if isinstance(key, ast.Constant) and key.value == model_type
    )
    mapping = ast.literal_eval(model_mapping)
    description = {f"model.layers.0.mlp.{name}.weight": "W4A8_DYNAMIC" for name in mapping["experts"]}
    namespace = {"MappingProxyType": MappingProxyType, "logger": Mock()}
    load_function(path, "_is_missing_v_shard", namespace)
    load_function(path, "get_linear_quant_type", namespace)
    lookup = load_function(path, "is_layer_skipped_ascend", namespace, "AscendModelSlimConfig")
    assert not lookup(SimpleNamespace(quant_description=description), "model.layers.0.mlp.experts", mapping)


def test_norm_bias_scan_is_reused_per_description():
    class CountedDict(dict):
        scans = 0

        def __iter__(self):
            self.scans += 1
            return super().__iter__()

    namespace = {"_NORM_BIAS_IN_QUANT_DESCRIPTION": {}}
    lookup = load_function("vllm_ascend/ops/layernorm.py", "_quant_description_has_norm_bias", namespace)
    description = CountedDict({"layer.norm.bias": "FLOAT"})
    assert lookup(description) and lookup(description)
    assert description.scans == 1
    assert not lookup({"layer.weight": "FLOAT"})


@pytest.mark.parametrize("layer_kind", ["linear", "moe", "embedding"])
@pytest.mark.parametrize("quant_type", ["W8A8_DYNAMIC", "FLOAT", None])
def test_quant_method_resolves_each_layer_once(monkeypatch, layer_kind, quant_type):
    class Linear:
        pass

    class Attention:
        pass

    class Embedding:
        pass

    class MoE:
        moe_config = object()

    adapters = ModuleType("vllm_ascend.quantization.method_adapters")
    for name in ("AscendLinearMethod", "AscendFusedMoEMethod", "AscendEmbeddingMethod", "AscendKVCacheMethod"):
        setattr(adapters, name, Mock())
    monkeypatch.setitem(sys.modules, adapters.__name__, adapters)
    linear_module = ModuleType("vllm_ascend.ops.linear")
    linear_module.AscendUnquantizedLinearMethod = Mock()
    monkeypatch.setitem(sys.modules, linear_module.__name__, linear_module)
    moe_module = ModuleType("vllm_ascend.ops.fused_moe.fused_moe")
    moe_module.AscendUnquantizedFusedMoEMethod = Mock()
    monkeypatch.setitem(sys.modules, moe_module.__name__, moe_module)

    resolve = Mock(return_value=quant_type)
    scheme = Mock()
    namespace = {
        "__name__": "vllm_ascend.quantization.modelslim_config",
        "__package__": "vllm_ascend.quantization",
        "get_current_vllm_config": lambda: SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="test"), hf_text_config=SimpleNamespace())
        ),
        "packed_modules_model_mapping": {},
        "Mapping": Mapping,
        "LinearBase": Linear,
        "AttentionLayerBase": Attention,
        "VocabParallelEmbedding": Embedding,
        "UnquantizedEmbeddingMethod": Mock(),
        "_is_fused_moe_layer": lambda layer: isinstance(layer, MoE),
        "get_linear_quant_type": resolve,
        "create_scheme_for_layer": scheme,
        "logger": Mock(),
    }
    method = load_function(
        "vllm_ascend/quantization/modelslim_config.py", "get_quant_method", namespace, "AscendModelSlimConfig"
    )
    config = SimpleNamespace(quant_description={}, packed_modules_mapping={}, quant_prefix_mapper=lambda _, name: name)
    method(config, {"linear": Linear, "moe": MoE, "embedding": Embedding}[layer_kind](), "layer")
    resolve.assert_called_once_with({}, "layer", {})
    if quant_type == "W8A8_DYNAMIC":
        scheme.assert_called_once_with({}, "layer", "moe" if layer_kind == "moe" else "linear", {}, quant_type)
    else:
        scheme.assert_not_called()


def test_resolved_quant_type_is_reused_by_scheme():
    resolve = Mock(side_effect=AssertionError("quant type was resolved twice"))
    scheme_class = Mock()
    namespace = {
        "logger": Mock(),
        "get_quant_type_for_layer": resolve,
        "get_scheme_class": Mock(return_value=scheme_class),
    }
    create = load_function("vllm_ascend/quantization/modelslim_config.py", "create_scheme_for_layer", namespace)
    assert create({}, "layer", "linear", {}, "W8A8_DYNAMIC") is scheme_class.return_value
    namespace["get_scheme_class"].assert_called_once_with("W8A8_DYNAMIC", "linear")
    resolve.assert_not_called()


def test_expert_map_cache_preserves_independent_copies():
    determine = Mock(return_value=(None, torch.arange(4), None))
    namespace = {"_LINEAR_EXPERT_MAP_CACHE": {}, "determine_expert_map": determine, "torch": torch}
    initialize = load_function("vllm_ascend/eplb/core/eplb_utils.py", "init_eplb_config", namespace)
    config = SimpleNamespace(expert_map_path=None, dynamic_eplb=False, num_redundant_experts=0)
    moe = SimpleNamespace(num_experts=4, ep_size=2, ep_rank=0)
    first = initialize(config, 0, moe)[1]
    first[0] = 99
    second = initialize(config, 1, moe)[1]
    assert second.tolist() == [0, 1, 2, 3]
    determine.assert_called_once_with(2, 0, 4)
