import gc
import logging
import re

import torch
import torch.nn as nn

from qdiff.base.base_quantizer import BaseQuantizer
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.lsgquant.lsgquant_quant_layer import LSGQuantizedLinear
from qdiff.quarot.quarot_quant_layer import QuarotQuantizedLinear
from qdiff.quarot.quarot_utils import hadamard_from_sign
from qdiff.smooth_quant.sq_quant_layer import SQQuantizedLinear
from qdiff.utils import apply_func_to_submodules
from qdiff.viditq.viditq_quant_layer import ViDiTQuantizedLinear

logger = logging.getLogger(__name__)

# Quantization methods released with LSGQuant, and the quantized layer each of
# them instantiates. A config without any of these keys falls back to plain
# MinMax quantization (``qdiff.base.quant_layer.QuantizedLinear``).
QUANT_METHOD_MAP = {
    "smooth_quant": SQQuantizedLinear,
    "quarot": QuarotQuantizedLinear,
    "viditq": ViDiTQuantizedLinear,
    "lsgquant": LSGQuantizedLinear,
}


def quant_layer_refactor_(
    submodule: nn.Linear,
    name: str,
    parent_module: nn.Module,
    quant_config,
    full_name: str,
    remain_fp_regex,
    device
):
    # 1. select the quantized layer required by the layer name
    quant_layer_type = QuantizedLinear
    for method_name, quant_class_name in QUANT_METHOD_MAP.items():
        if quant_config.get(method_name):
            layer_regex = quant_config[method_name].layer_name_regex
            match = re.search(re.compile(layer_regex), full_name)
            if match:
                quant_layer_type = quant_class_name
                logger.info('[INFO] setting {} for layer {}'.format(method_name, full_name))

    # 2. keep some layers in full precision (fixed), feed in from config
    # set some layers as FP (fixed), feed in from config
    if remain_fp_regex is not None:
        pattern = re.compile(remain_fp_regex)
        if pattern.search(full_name):
            logger.info(f"remain {full_name} quant as FP due to fp_regex")
            return

    # 3. set the quantizer and the module name
    in_features = submodule.in_features
    out_features = submodule.out_features
    bias = (submodule.bias is not None)
    new_quant_layer = quant_layer_type(
        in_features,out_features,bias,device,quant_config,submodule
    )
    setattr(parent_module, name, new_quant_layer)

    # set the module_name for quant_layer and quantizers
    setattr(getattr(parent_module, name), 'module_name', full_name)
    if getattr(parent_module, name).w_quantizer is not None:
        setattr(getattr(parent_module, name).w_quantizer, 'module_name', full_name)
    if getattr(parent_module, name).a_quantizer is not None:
        setattr(getattr(parent_module, name).a_quantizer, 'module_name', full_name)


def bitwidth_refactor_(submodule, name, parent_module, quant_config, full_name):
    layer_regex_list_w = quant_config.mixed_precision.weight.layer_name_regex
    layer_regex_list_a = quant_config.mixed_precision.act.layer_name_regex

    for idx, layer_regex in enumerate(layer_regex_list_w):
        if len(layer_regex) == 0: # 0 being empty, skip
            continue
        else:
            match = re.search(re.compile(layer_regex), full_name)
            if match:
                if idx == 0: # the FP16
                    submodule.quant_mode = False
                    logger.info(f'[Mixed Precision] set the {full_name} W as FP16')
                else:
                    submodule.w_quantizer.bitwidth_refactor(idx-1)
                    logger.info(f'[Mixed Precision] set the {full_name} W as {submodule.w_quantizer.bitwidth_list[idx-1]} bit')

    for idx, layer_regex in enumerate(layer_regex_list_a):
        if len(layer_regex) == 0: # 0 being empty, skip
            continue
        else:
            match = re.search(re.compile(layer_regex), full_name)
            if match:
                if idx == 0: # the FP16
                    submodule.quant_mode = False
                    logger.info(f'[Mixed Precision] set the {full_name} A as FP16')
                else:
                    submodule.a_quantizer.bitwidth_refactor(idx-1)
                    logger.info(f'[Mixed Precision] set the {full_name} A as {submodule.a_quantizer.bitwidth_list[idx-1]} bit')


def restore_rotation_matrix_(parent_module, quant_param_dict, full_name, device):
    """Rebuild ``parent_module.rotation_matrix`` for rotation-based quantizers.

    Three cases, in order of preference:

    1. legacy parameter files: a dense ``rotation_matrix`` is stored -> use it;
    2. released compact files: the +/-1 sign vector ``had_sign`` is stored ->
       rebuild ``H = matmul_hadU(diag(had_sign))`` bit-exactly;
    3. parameter files that do not describe the rotation at all -> sample a new one.
    """
    entry = quant_param_dict[full_name]
    if entry.get('rotation_matrix', None) is not None:
        parent_module.rotation_matrix = entry['rotation_matrix'].to(device)
    elif entry.get('had_sign', None) is not None:
        parent_module.had_sign = entry['had_sign']
        parent_module.rotation_matrix = hadamard_from_sign(entry['had_sign']).to(device)
    else:
        parent_module.get_rotation_matrix(device)


def load_quant_param_dict_(submodule, full_name, parent_module, quant_param_dict, model, device):
    submodule.delta = quant_param_dict[full_name]['delta']
    submodule.zero_point = quant_param_dict[full_name]['zero_point']
    model.quant_param_dict[full_name] = quant_param_dict[full_name]

    has_channel_mask = hasattr(parent_module, 'channel_mask')
    has_rotation_matrix = hasattr(parent_module, 'rotation_matrix')
    has_L1L2 = hasattr(parent_module, 'L1') and hasattr(parent_module, 'L2')

    # viditq: channel_mask + rotation_matrix
    # quarot: rotation_matrix
    # smoothquant: channel_mask
    # lsgquant: rotation_matrix + L1 + L2
    if has_channel_mask and has_rotation_matrix and not has_L1L2:  # viditq
        assert isinstance(parent_module, ViDiTQuantizedLinear)
        restore_rotation_matrix_(parent_module, quant_param_dict, full_name, device)
        parent_module.channel_mask = quant_param_dict[full_name]['channel_mask'].to(device)
        parent_module.update_quantized_weight_rotated_and_scaled()
    elif not has_channel_mask and has_rotation_matrix and not has_L1L2:  # quarot
        assert isinstance(parent_module, QuarotQuantizedLinear)
        restore_rotation_matrix_(parent_module, quant_param_dict, full_name, device)
        parent_module.update_quantized_weight_rotated()
    elif has_channel_mask and not has_rotation_matrix and not has_L1L2:  # smoothquant
        assert isinstance(parent_module, SQQuantizedLinear)
        parent_module.channel_mask = quant_param_dict[full_name]['channel_mask'].to(device)
        parent_module.update_quantized_weight_scaled()
    elif not has_channel_mask and has_rotation_matrix and has_L1L2:  # lsgquant
        assert isinstance(parent_module, LSGQuantizedLinear)
        restore_rotation_matrix_(parent_module, quant_param_dict, full_name, device)
        parent_module.L1 = quant_param_dict[full_name]['L1'].to(device)
        parent_module.L2 = quant_param_dict[full_name]['L2'].to(device)
        parent_module.reload_R_quantized()
    elif isinstance(parent_module, QuantizedLinear):  # min_max / base
        # Nothing else to rebuild: the weight is quantized when the layer is
        # refactored and the activations are quantized online, so the delta /
        # zero_point assigned above are all the layer needs.
        pass
    else:
        logger.warning(f'no restoring rule for layer {full_name}, only delta / zero_point are loaded')

    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f'Loading layer {full_name}')


def save_quant_param_dict_(submodule, full_name, parent_module, model):
    model.quant_param_dict[full_name] = {}
    model.quant_param_dict[full_name]['delta'] = submodule.delta
    model.quant_param_dict[full_name]['zero_point'] = submodule.zero_point

    if getattr(parent_module, 'had_sign', None) is not None:
        model.quant_param_dict[full_name]['had_sign'] = parent_module.had_sign
    elif getattr(parent_module, 'rotation_matrix', None) is not None:
        model.quant_param_dict[full_name]['rotation_matrix'] = parent_module.rotation_matrix

    attrs_maybe_saved = ['channel_mask', 'lora_weight', 'lora_bias', 'L1', 'L2']

    for attr in attrs_maybe_saved:
        if hasattr(parent_module, attr):
            model.quant_param_dict[full_name][attr] = getattr(parent_module, attr)


def set_init_done_(submodule):
    submodule.init_done = True
