import os
import json
import torch
import pyiqa
import numpy as np
from tqdm import tqdm

from eval_utils import (
    load_sequence, 
    crop_border,
    rgb_to_y,
    fr_metrics,
)

def rgb_to_grayscale(tensor):
    # RGB tensor -> grayscale
    r, g, b = tensor[0:1, :, :], tensor[1:2, :, :], tensor[2:3, :, :]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray

def calculate_ncc(gt_patch, pred_frame):
    """
    Normalized cross correlation (NCC), used as the similarity of two frames.
    """
    # grayscale
    gt_gray = rgb_to_grayscale(gt_patch)
    pred_gray = rgb_to_grayscale(pred_frame)

    # normalize
    gt_gray = (gt_gray - gt_gray.mean()) / (gt_gray.std() + 1e-8)
    pred_gray = (pred_gray - pred_gray.mean()) / (pred_gray.std() + 1e-8)

    # cross correlation
    return torch.sum(gt_gray * pred_gray) / (gt_gray.numel())

def find_best_crop_position(gt_frames, pred_frames, sample_frames=5):
    """
    Return the crop position of the GT frames with the highest mean NCC.
    """
    # sizes
    _, _, h_g, w_g = gt_frames.shape
    _, _, h_p, w_p = pred_frames.shape

    # every candidate crop position
    h_positions = h_g - h_p + 1
    w_positions = w_g - w_p + 1

    # sample a few frames instead of evaluating all of them
    frame_indices = torch.linspace(0, gt_frames.shape[0]-1, sample_frames).long()

    # mean NCC of every position
    ncc_scores = torch.zeros(h_positions, w_positions)

    # score every position
    for ij in tqdm(range(h_positions*w_positions), desc="Evaluating crop positions"):
        i,j = ij // w_positions, ij % w_positions
        total_ncc = 0.0
        count = 0

        for idx in frame_indices:
            gt_frame = gt_frames[idx]
            pred_frame = pred_frames[idx]

            # crop the GT frame at this position
            gt_patch = gt_frame[:, i:i+h_p, j:j+w_p]

            # similarity of the cropped pair
            ncc = calculate_ncc(gt_patch, pred_frame)
            total_ncc += ncc
            count += 1

        ncc_scores[i, j] = total_ncc / count

    # best position
    # print(ncc_scores)
    # time.sleep(1)
    best_i, best_j = np.unravel_index(torch.argmax(ncc_scores).item(), ncc_scores.shape)

    return best_i, best_j

def crop_img_custom(img, top, left, target_h, target_w):
    return img[:, top:top+target_h, left:left+target_w]

def crop_img_center(img, target_h, target_w):
    _, h, w = img.shape
    top = max((h - target_h) // 2, 0)
    left = max((w - target_w) // 2, 0)
    return img[:, top:top+target_h, left:left+target_w]

def crop_img_top_left(img, target_h, target_w):
    # Crop image from top-left corner to (target_h, target_w)
    return img[:, :target_h, :target_w]

def match_resolution(
    gt_frames: torch.Tensor, 
    pred_frames: torch.Tensor, 
    method: str = 'top-left', 
    name: str = None, 
    sample_frames: int = 5
):
    """Crop the GT and the prediction to a common resolution.

    Args:
        gt_frames: torch.Tensor [T, C, H, W]
        pred_frames: torch.Tensor [T, C, h, w]
        method: str, 'center', 'top-left' or 'auto'
        name: str, clip name, only used for the log messages
        sample_frames: int, frames evaluated when method='auto'

    Returns:
        gt_frames, pred_frames: the cropped clips
    """
    t = min(gt_frames.shape[0], pred_frames.shape[0])
    gt_frames = gt_frames[:t]
    pred_frames = pred_frames[:t]
    _, _, h_g, w_g = gt_frames.shape
    _, _, h_p, w_p = pred_frames.shape

    target_h = min(h_g, h_p)
    target_w = min(w_g, w_p)

    if (h_g != h_p or w_g != w_p) and name:
        tqdm.write(f"[{name}] Resolution mismatch detected: GT is ({h_g}, {w_g}), Pred is ({h_p}, {w_p}).")
        if method == 'auto':
            # search the crop position of the GT clip with the highest NCC
            tqdm.write(f"[{name}] Finding best crop position using NCC...")
            top, left = find_best_crop_position(gt_frames, pred_frames, sample_frames)
            tqdm.write(f"[{name}] Best crop position found: top={top}, left={left}")
            # apply the position found to every GT frame
            gt_frames = torch.stack([crop_img_custom(f, top, left, target_h, target_w) for f in gt_frames])
            pred_frames = torch.stack([crop_img_top_left(f, target_h, target_w) for f in pred_frames])
        elif method == 'center':
            tqdm.write(f"[{name}] Both GT and Pred were center cropped to ({target_h}, {target_w}).")
            gt_frames = torch.stack([crop_img_center(f, target_h, target_w) for f in gt_frames])
            pred_frames = torch.stack([crop_img_center(f, target_h, target_w) for f in pred_frames])
        elif method == 'top-left':  # top-left
            tqdm.write(f"[{name}] Both GT and Pred were top-left cropped to ({target_h}, {target_w}).")
            gt_frames = torch.stack([crop_img_top_left(f, target_h, target_w) for f in gt_frames])
            pred_frames = torch.stack([crop_img_top_left(f, target_h, target_w) for f in pred_frames])
        else:
            raise ValueError(f"not supported method: {method}")

    return gt_frames, pred_frames


def init_models(metrics: list[str], device: torch.device):
    """Build the pyiqa metrics.

    Args:
        metrics (list[str]): metrics to evaluate
        device (torch.device): device of the metrics

    Returns:
        models (dict{str, InferenceModel}): metric name -> pyiqa model
    """
    models = {}
    for name in metrics:
        try:
            models[name] = pyiqa.create_metric(name, device = device).eval()
        except Exception as e:
            print(f"Failed to initialize metric '{name}': {e}")
    return models

def compute_metrics(pred_frames, gt_frames, models, device, batch_mode, crop, test_y_channel):
    if batch_mode:
        pred_batch = pred_frames.to(device)  # [F, C, H, W]
        gt_batch = gt_frames.to(device)      # [F, C, H, W]

        results = {}
        for name, model in models.items():
            if name in fr_metrics:
                pred_eval = pred_batch
                gt_eval = gt_batch
                if crop > 0:
                    pred_eval = crop_border(pred_eval, crop)
                    gt_eval = crop_border(gt_eval, crop)
                if test_y_channel:
                    pred_eval = rgb_to_y(pred_eval)
                    gt_eval = rgb_to_y(gt_eval)
                values = model(pred_eval, gt_eval)  # [F]
            else:
                values = model(pred_batch)  # no-reference
            results[name] = round(values.mean().item(), 4)
        return results

    else:
        results = {name: [] for name in models}
        for pred, gt in zip(pred_frames, gt_frames):
            pred = pred.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)

            for name, model in models.items():
                if name in fr_metrics:
                    pred_eval = pred
                    gt_eval = gt
                    if crop > 0:
                        pred_eval = crop_border(pred_eval, crop)
                        gt_eval = crop_border(gt_eval, crop)
                    if test_y_channel:
                        pred_eval = rgb_to_y(pred_eval)
                        gt_eval = rgb_to_y(gt_eval)
                    value = model(pred_eval, gt_eval).item()
                else:
                    value = model(pred).item()
                results[name].append(value)

        return {k: round(np.mean(v), 4) for k, v in results.items()}


def process(gt_root, pred_root, out_path, metrics, batch_mode, crop, test_y_channel, is_center, device):
    print(f"Using device: {device}")
    models = init_models(metrics, device)

    # list the GT and the prediction files
    has_gt = bool(gt_root and os.path.exists(gt_root))

    if has_gt:
        gt_files = {os.path.splitext(f)[0]: os.path.join(gt_root, f) for f in os.listdir(gt_root)}
    pred_files = {os.path.splitext(f)[0]: os.path.join(pred_root, f) for f in os.listdir(pred_root)}

    pred_names = sorted(pred_files.keys())
    results = {}
    aggregate = {metric: [] for metric in metrics}

    for name in tqdm(pred_names, desc="Evaluating"):
        # # valida
        # name_hr = name.replace('_CAT_A_x4', '').replace('img_', 'img')
        name_hr = name
        if has_gt and name_hr not in gt_files:
            print(f"Skipping {name_hr}: no matching GT file.")
            continue

        pred_path = pred_files[name]
        gt_path = gt_files[name_hr] if has_gt else None

        try:
            # read the prediction
            pred_frames = load_sequence(pred_path)

            # align the pair and compute the metrics
            if has_gt:
                gt_frames = load_sequence(gt_path)
                gt_frames, pred_frames = match_resolution(gt_frames, pred_frames, name=name, sample_frames=8, method='top-left')
                scores = compute_metrics(pred_frames, gt_frames, models, device, batch_mode, crop, test_y_channel)
            else:
                nr_models = {k: v for k, v in models.items() if k not in fr_metrics}
                if not nr_models:  # no metric can be computed without GT
                    print(f"Skipping {name}: GT is not provided and no NR-IQA metrics found.")
                    continue
                dummy_gt = pred_frames
                scores = compute_metrics(pred_frames, dummy_gt, nr_models, device, batch_mode, crop, test_y_channel)

            results[name] = scores
            for k in scores:
                aggregate[k].append(scores[k])
        except Exception as e:
            print(f"Error processing {name}: {e}")

    # report and save
    print("\nPer-sample Results:")
    for name in sorted(results):
        print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in results[name].items()))

    print("\nOverall Average Results:")
    count = len(results)
    if count > 0:
        overall_avg = {k: round(np.mean(v), 4) for k, v in aggregate.items()}
        for k, v in overall_avg.items():
            print(f"{k.upper()}: {v:.4f}")
    else:
        overall_avg = {}
        print("No valid samples were processed.")

    print(f"\nProcessed {count} samples.")

    output = {
        "per_sample": results,
        "average": overall_avg,
        "count": count
    }

    os.makedirs(out_path, exist_ok=True)
    out_name = 'metrics_'
    for metric in metrics:
        out_name += f"{metric}_"
    out_name = out_name.rstrip('_') + '.json'
    out_path = os.path.join(out_path, out_name)

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='', help='Path to GT folder (optional for NR-IQA)')
    parser.add_argument('--pred', type=str, required=True, help='Path to predicted results folder')
    parser.add_argument('--out', type=str, default='', help='Path to save JSON output (as directory)')
    parser.add_argument('--metrics', type=str, default='psnr,ssim,clipiqa',
                        help='Comma-separated list of metrics: psnr,ssim,clipiqa,lpips,...')
    parser.add_argument('--batch_mode', action='store_true', help='Use batch mode for metrics computation')
    parser.add_argument('--crop', type=int, default=0, help='Crop border size for PSNR/SSIM')
    parser.add_argument('--test_y_channel', action='store_true', help='Use Y channel for PSNR/SSIM')
    parser.add_argument('--is_center', action='store_true', help='Use center crop for PSNR/SSIM')

    args = parser.parse_args()

    # where to write the results
    out = args.pred if args.out == '' else args.out

    # metric string -> list
    metric_list = [m.strip().lower() for m in args.metrics.split(',')]

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    process(args.gt, args.pred, out, metric_list, args.batch_mode, args.crop, args.test_y_channel, args.is_center, device)
