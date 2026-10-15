import os
import json
import shutil

from eval_utils import is_video_file
from eval_metric.RAFT.ewarp import Ewarp

def process(pred_root, out_path, args):
    all_items = os.listdir(pred_root)
    videos_count = sum(1 for item in all_items if is_video_file(os.path.join(pred_root, item)))
    print(f"Found {videos_count} videos.")

    pred_files = {
        os.path.splitext(item)[0]: os.path.join(pred_root, item) 
        for item in all_items 
        if is_video_file(os.path.join(pred_root, item))
    }

    if not pred_files:
        print("No valid folders or video files found in the specified directory.")
        return

    # E*_warp reads a flat directory of videos, copy the predictions there
    os.makedirs(out_path, exist_ok=True)
    temp_video_path = os.path.join(out_path, "temp")
    os.makedirs(temp_video_path, exist_ok=True)
    for name, video_path in pred_files.items():
        new_video_path = os.path.join(temp_video_path, f"{name}.mp4")
        shutil.copy(video_path, new_video_path)

    results, avg_score = Ewarp(args, temp_video_path)
    shutil.rmtree(temp_video_path)

    count = len(results)
    if count > 0:
        overall_avg = avg_score
        print(results)
    else:
        overall_avg = None
        print("No valid samples were processed.")

    print(f"\nProcessed {count} samples.")
    print(f"Average score: {overall_avg:.4f}")
    output = {
        "per_sample": results,
        "average": round(overall_avg, 4),
        "count": count
    }

    file_out_path = os.path.join(out_path, 'metrics_ewarp.json')
    with open(file_out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {file_out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default='', help="Specify the path of generated videos")
    parser.add_argument('--out', type=str, default='', help='Path to save JSON output (as directory)')

    parser.add_argument("--metric", type=str, default='warping_error', help="Specify the metric to be used")
    parser.add_argument('--model', type=str, default='eval_metric/RAFT/models/raft-things.pth',help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    args = parser.parse_args()
    out = args.pred if args.out == '' else args.out

    process(args.pred, out, args)
