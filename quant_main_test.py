import torch
import os
import logging

from qdiff.utils import (
    seed_everything, 
    setup_logging
)

from omegaconf import OmegaConf
from main_test import WanValidationModule
from customize_model import CustomizeWanModel
from trainer.trainer import launch_validation_task_sr
from datasets.datasets import RealSRValDataset
from trainer.parser import wan_parser

def main(args):
    seed_everything(args.seed)
    torch.set_grad_enabled(False)

    if (args.log is not None) and (not os.path.exists(args.log)):
        os.makedirs(args.log)
    log_file = os.path.join(args.log, 'quant_inference.log')
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    val_dataset = RealSRValDataset(
        lq_root=os.path.join(args.val_path, 'LQ-Video'),
        hq_root=os.path.join(args.val_path, 'GT-Video'),
        prompt_cache=args.prompt_cache,
    )

    # load models and change dit
    whole_model = WanValidationModule(
        args=args,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
    ) 

    # ---- assign quant configs ------
    quant_config = OmegaConf.load(args.quant_config)
    model: CustomizeWanModel = whole_model.pipe.dit
    model.convert_quant(quant_config)

    quant_param_ckpt = torch.load(
        os.path.join(args.log, quant_config.quant_param.save_path), 
        weights_only=True
    )
    model.load_quant_param_dict(quant_param_ckpt)
    model.set_init_done()
    logger.info(str(model))
    # if args.verbose: exit()

    launch_validation_task_sr(whole_model, val_dataset, output_path=args.output_path, args=args)

if __name__ == "__main__":
    parser = wan_parser()

    # quant params
    parser.add_argument("--log", type=str)
    parser.add_argument('--quant_config', required=True, type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hardware", action='store_true', help='whether to use_cuda_kernel')
    parser.add_argument("--quant_weight_ckpt", type=str, default=None)

    # extra params
    parser.add_argument("--verbose", action="store_true", help="Debug mode for check models")

    args = parser.parse_args()
    main(args)
