"""Inference / calibration loops used by LSGQuant.

Only the validation (FP inference) and calibration-data loops are kept in the
released code: LSGQuant is a *post-training* quantization method and the FP
backbone is released as weights (see README).
"""

import os
import torch
from tqdm import tqdm

from trainer.utils import save_video_with_imageio

def launch_validation_task_sr(model, val_dataset=None, output_path="./models", args=None):
    model.eval()
    os.makedirs(output_path, exist_ok=True)
    with torch.no_grad():
        for data in tqdm(val_dataset):
            name = data["name"][:-4] + '.mp4'
            pred_video = model(data)
            if not args.debug:
                save_video_with_imageio(pred_video, os.path.join(output_path, name))

def launch_validation_task_sr_calib(model, val_dataset=None):
    """Single forward pass over the calibration set (used to collect statistics)."""
    model.eval()
    with torch.no_grad():
        for data in tqdm(val_dataset):
            _ = model.inference(data)
