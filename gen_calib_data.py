"""Collect the calibration statistics used by the quantization methods.

Two modes are supported:

* ``--calib_mode var`` (default): the layer sensitivity used by VOLTS. 
  For every linear layer of the DiT we record the per-channel
  variance of its input on each calibration sample; the scalar stored for a
  layer is the mean over channels of the variance of those values across the
  calibration samples. ``ptq.py`` turns it into a per-layer iteration budget
  through the two thresholds ``low_threshold`` / ``high_threshold``.
* ``--calib_mode act_max``: the per-channel activation (absolute) maximum used
  by the SmoothQuant / ViDiT-Q baselines, stored as ``[num_forward_passes, C]``.

The output file name is taken from ``calib_data.save_path`` of the quantization
config, and is written into ``--log`` (e.g. ``calib``).
"""

import logging
import os

import torch
from omegaconf import OmegaConf

from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging

from datasets.datasets import RealSRDataset
from hooks import SaveActivationMaxHook, SaveActivationVarHook, add_hook_to_module_
from trainer.parser import wan_parser
from trainer.trainer import launch_validation_task_sr_calib
from wan_training_module import WanTrainingModule


def main(args):
    seed_everything(args.seed)
    torch.set_grad_enabled(False)

    if (args.log is not None) and (not os.path.exists(args.log)):
        os.makedirs(args.log)

    log_file = os.path.join(args.log, 'gen_calib_data.log')
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    model = WanTrainingModule(
        args=args,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
    ).to("cuda")

    val_dataset = RealSRDataset(
        data_root=args.dataset_base_path,
        video_column=args.dataset_metadata_path,
        max_num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        degradation_config=args.degradation_config,
        prompt_cache=args.prompt_cache,
        nums=args.num_videos
    )

    dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True,
        pin_memory_device="cuda"
    )

    quant_config = OmegaConf.load(args.quant_config)

    # INFO: add the hook for hooking the activations
    if args.calib_mode == "var":
        hook_cls = SaveActivationVarHook                 # VOLTS
    elif args.calib_mode == "act_max":
        hook_cls = SaveActivationMaxHook                 # SmoothQuant / ViDiT-Q baselines
    else:
        raise ValueError(f"unknown --calib_mode {args.calib_mode}")

    hook_d = apply_func_to_submodules(
        model.pipe.dit.blocks,
        class_type=torch.nn.Linear,  # add hook to all objects of this cls
        function=add_hook_to_module_,
        return_d={},
        hook_cls=hook_cls,
    )

    launch_validation_task_sr_calib(model, dataloader)

    save_d = {}
    for k, v in hook_d.items():
        if args.calib_mode == "var":
            vout = torch.stack(v.outputs, dim=0)   # [N_calib_samples, C]
            save_d[k] = vout.var(dim=0).mean().item()   # VOLTS layer sensitivity
            logger.info(f'layer_name: {k}, value: {save_d[k]}')
        else:
            vout = torch.stack(v.outputs, dim=0)   # [N_forward_passes, C]
            save_d[k] = vout
            logger.info(f'layer_name: {k}, shape: {tuple(save_d[k].shape)}')
        v.hook_handle.remove()

    out_path = os.path.join(args.log, quant_config.calib_data.save_path)
    torch.save(save_d, out_path)
    logger.info(f'saved calib data in {out_path}')


if __name__ == "__main__":
    parser = wan_parser()
    parser.add_argument("--log", type=str)
    parser.add_argument('--quant_config', required=True, type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--calib_mode", type=str, default="var", choices=["var", "act_max"],
        help="'var': VOLTS layer-sensitivity statistic; 'act_max': per-channel activation max for the baselines",
    )
    parser.add_argument(
        "--num_videos", type=int, default=50,
        help="Number of calibration videos sampled from --dataset_base_path.",
    )
    args = parser.parse_args()
    main(args)
