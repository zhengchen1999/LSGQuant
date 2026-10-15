"""Weight / activation quantization parameter estimation (PTQ).

The PTQ pipeline of LSGQuant is:

  1. ``gen_calib_data.py`` collects the calibration statistics,
  2. this script builds the quantization parameters of every DiT linear layer,
  3. ``quant_main_test.py`` runs the quantized inference.
"""

import logging
import os

import torch
from omegaconf import OmegaConf

from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging

from customize_model import CustomizeWanModel
from main_test import WanValidationModule
from trainer.parser import wan_parser


def main(args):
    seed_everything(args.seed)
    torch.set_grad_enabled(False)

    if (args.log is not None) and (not os.path.exists(args.log)):
        os.makedirs(args.log)
    log_file = os.path.join(args.log, 'ptq.log')
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    # load models and change dit
    whole_model = WanValidationModule(
        args=args,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
    )

    # ---- assign quant configs ------
    quant_config = OmegaConf.load(args.quant_config)
    print(quant_config)
    model: CustomizeWanModel = whole_model.pipe.dit
    model.convert_quant(quant_config)

    # INFO: The PTQ process:
    # for simple PTQ with dynamic act quant (MinMax):
    # the weight are quantized with quant_model initialization.
    # the act quant params are calculated online.
    calib_data = None
    if quant_config.get("calib_data", None) is not None:
        calib_data_path = os.path.join(args.calib_dir, quant_config.calib_data.save_path)
        calib_data = torch.load(calib_data_path, weights_only=True)
        logger.info(f'loaded calib data from {calib_data_path}')

    if quant_config.get("smooth_quant", None) is not None:
        # ---- SmoothQuant ------------------------------------------------------
        # Channel-wise scaling migrates the activation outliers into the weights,
        # using the per-channel activation maxima collected by
        # `gen_calib_data.py --calib_mode act_max`.
        from qdiff.smooth_quant.sq_quant_layer import SQQuantizedLinear

        def init_sq_channel_mask_(module, full_name: str, calib_data):
            print(f"smooth_quant method initializing: {full_name}")
            assert isinstance(module, SQQuantizedLinear)
            calib_name = full_name.removeprefix("blocks.")
            act_mask = calib_data[calib_name].max(dim=0)[0].to("cuda")   # [T, C] -> [C]
            module.get_channel_mask(act_mask)  # set self.channel_mask
            module.update_quantized_weight_scaled()

        assert calib_data is not None, \
            "SmoothQuant needs calibration data, run gen_calib_data.py --calib_mode act_max first"
        apply_func_to_submodules(
            model,
            class_type=SQQuantizedLinear,  # add hook to all objects of this cls
            function=init_sq_channel_mask_,
            calib_data=calib_data,
            full_name='',
        )

    elif quant_config.get("quarot", None) is not None:
        # ---- QuaRot -----------------------------------------------------------
        # The rotation is applied while loading the quantization parameters
        # (qdiff/base/quant_model.py), so nothing has to be prepared here.
        pass

    elif quant_config.get("viditq", None) is not None:
        # ---- ViDiT-Q ----------------------------------------------------------
        from qdiff.viditq.viditq_quant_layer import ViDiTQuantizedLinear

        def init_rotation_and_channel_mask_(module, full_name: str, calib_data):
            print(f"viditq method initializing: {full_name}")
            assert isinstance(module, ViDiTQuantizedLinear)
            calib_name = full_name.removeprefix("blocks.")
            act_mask = calib_data[calib_name].max(dim=0)[0].to("cuda")  # [T, C] -> [C]
            module.get_channel_mask(act_mask)  # set self.channel_mask
            module.get_rotation_matrix("cuda")
            module.update_quantized_weight_rotated_and_scaled()

        assert calib_data is not None, \
            "ViDiT-Q needs calibration data, run gen_calib_data.py --calib_mode act_max first"
        apply_func_to_submodules(
            model,
            class_type=ViDiTQuantizedLinear,  # add hook to all objects of this cls
            function=init_rotation_and_channel_mask_,
            full_name='',
            calib_data=calib_data,
        )

    elif quant_config.get("lsgquant") is not None:
        # ---- LSGQuant --------------------------------------------------
        from qdiff.lsgquant.lsgquant_quant_layer import LSGQuantizedLinear

        # VOLTS: Variance-Oriented Layer Training Strategy.
        # The layer sensitivity sigma_l^2 estimated on the calibration set splits the layers into three tiers, 
        # each of which gets a different number of QAO iterations:
        #
        #   sigma_l^2 <  delta_1 : low variance, frozen after initialization
        #   delta_1 <= s^2 < delta_2 : medium variance, lightly adapted
        #   sigma_l^2 >= delta_2 : high variance, fully optimized
        #
        # Default values are the values used in the paper.
        lsg_config = quant_config.lsgquant
        low_threshold = float(lsg_config.get("low_threshold", 1e-3))     # delta_1
        high_threshold = float(lsg_config.get("high_threshold", 0.075))  # delta_2
        iter_frozen = int(lsg_config.get("iter_frozen", 1))
        iter_light = int(lsg_config.get("iter_light", 30))
        iter_converged = int(lsg_config.get("iter_converged", 1000))

        def init_rotation_and_qao_(module, full_name: str, rank: int, calib_data, max_iter=None):
            print(f"lsgquant method initializing: {full_name}")
            assert isinstance(module, LSGQuantizedLinear)
            module.get_rotation_matrix("cuda")

            if calib_data is not None:
                calib_name = full_name.removeprefix("blocks.")
                sensitivity = float(calib_data[calib_name])
                if sensitivity < low_threshold:
                    max_iter = iter_frozen
                elif sensitivity < high_threshold:
                    max_iter = iter_light
                else:
                    max_iter = iter_converged
                print(f"  VOLTS: sensitivity={sensitivity:.6f} -> {max_iter} QAO iterations")

            # QAO: Quantization-Aware Alternating Optimization
            best_cnt, best_error = module.update_quantized_weight_rotated_and_svd(
                rank=rank, max_iter=max_iter
            )
            return best_cnt, best_error

        if calib_data is not None:
            apply_func_to_submodules(
                model,
                class_type=LSGQuantizedLinear,  # add hook to all objects of this cls
                function=init_rotation_and_qao_,
                full_name='',
                rank=lsg_config.rank,
                calib_data=calib_data,
            )
        else:
            # fixed-iteration ablation: every layer gets `lsgquant.iter` QAO rounds
            apply_func_to_submodules(
                model,
                class_type=LSGQuantizedLinear,  # add hook to all objects of this cls
                function=init_rotation_and_qao_,
                full_name='',
                rank=lsg_config.rank,
                max_iter=lsg_config.get("iter", 30),
                calib_data=None,
            )

    else:
        # ---- MinMax (the "base" configuration) --------------------------------
        # Weights are quantized when the layer is refactored and the activation
        # quantization parameters are computed online, so no calibration is needed.
        pass

    model.set_init_done()
    model.save_quant_param_dict()
    out_path = os.path.join(args.log, quant_config.quant_param.save_path)
    torch.save(model.quant_param_dict, out_path)
    logger.info(f'saved quant params into {out_path}')


if __name__ == "__main__":
    parser = wan_parser()
    parser.add_argument("--log", type=str)
    parser.add_argument(
        "--calib_dir", type=str, default=None,
        help="Directory holding the calibration data written by gen_calib_data.py "
             "(defaults to --log).",
    )
    parser.add_argument('--quant_config', required=True, type=str)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.calib_dir is None:
        args.calib_dir = args.log
    main(args)
