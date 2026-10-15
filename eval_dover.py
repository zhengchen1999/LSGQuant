import os
import numpy as np
import json
import shutil

from eval_utils import is_video_file, to_py
from eval_metric.DOVER.evaluate_a_set_of_videos import evaluate_set as dover

def process(pred_root, out_path):
    # count the video files of the prediction directory
    all_items = os.listdir(pred_root)
    videos_count = sum(1 for item in all_items if is_video_file(os.path.join(pred_root, item)))
    print(f"Found {videos_count} videos.")

    pred_files = {
        os.path.splitext(item)[0]: os.path.join(pred_root, item) 
        for item in all_items 
        if is_video_file(os.path.join(pred_root, item))
    }

    # nothing to evaluate
    if not pred_files:
        print("No valid folders or video files found in the specified directory.")
        return

    # DOVER reads a flat directory of videos, copy the predictions there
    os.makedirs(out_path, exist_ok=True)
    temp_video_path = os.path.join(out_path, "temp")
    os.makedirs(temp_video_path, exist_ok=True)
    for name, video_path in pred_files.items():
        new_video_path = os.path.join(temp_video_path, f"{name}.mp4")
        shutil.copy(video_path, new_video_path)

    # DOVER scores, one per video
    dover_results = dover(temp_video_path)
    shutil.rmtree(temp_video_path)
    results = {
        os.path.basename(full_video_name_list[0]): score
        for (full_video_name_list, score) in dover_results
    }

    # aggregate
    count = len(results)
    if count > 0:
        overall_avg = np.mean([_ for _ in results.values()])
        print(results)
    else:
        overall_avg = {}
        print("No valid samples were processed.")

    print(f"\nProcessed {count} samples.")
    print(f"Average score: {overall_avg:.4f}")
    output = {
        "per_sample": results,
        "average": round(float(overall_avg), 4),
        "count": count
    }

    file_out_path = os.path.join(out_path, 'metrics_dover.json')
    with open(file_out_path, 'w') as f:
        json.dump(to_py(output), f, indent=2)

    print(f"Results saved to: {file_out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True, help='Path to predicted results folder')
    parser.add_argument('--out', type=str, default='', help='Path to save JSON output (as directory)')
    args = parser.parse_args() 
    out = args.pred if args.out == '' else args.out
    process(args.pred, out)
