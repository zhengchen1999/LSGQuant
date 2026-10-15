import os
import subprocess
import re
import json

def evaluate_video(video_path, opt_path="./dover.yml", device="cuda", fusion=False):
    """
    Evaluate the quality score of one video.

    Args:
        video_path (str): path of the video file
        opt_path (str): path of the DOVER config file
        device (str): 'cuda' or 'cpu'
        fusion (bool): whether to report the fused score as well

    Returns:
        dict: the parsed evaluation result
    """
    # the video has to exist
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video file not found: {video_path}")

    # the config has to exist
    if not os.path.exists(opt_path):
        raise FileNotFoundError(f"config file not found: {opt_path}")

    # command line of the DOVER entry point
    cmd = [
        "python", 
        "evaluate_one_video.py",
        "-v", video_path,
        "-o", opt_path,
        "-d", str(device)
    ]
    
    if fusion:
        cmd.append("-f")
    
    # run and capture the output
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"evaluation failed with code {e.returncode}: {e.stderr}")

    # parse the output
    return _parse_output(output, video_path, fusion)

def _parse_output(output, video_path, fusion):
    """Parse the output of the evaluation script into a structured result."""
    result = {
        "video_path": video_path,
        "raw_output": output
    }

    if fusion:
        # fused score
        fusion_match = re.search(r"Normalized fused overall score \(scale in \[0,1\]\): ([-+]?\d*\.\d+|\d+)", output)
        if fusion_match:
            result["fused_score"] = float(fusion_match.group(1))
    else:
        # technical and aesthetic scores
        result["dataset_comparisons"] = {}

        # one block per reference dataset used for the normalization
        dataset_patterns = [
            (r"Compared with all videos in the ([\w_-]+) dataset:", "dataset"),
            (r"the technical quality of video .+ is better than (\d+)% of videos, with normalized score ([-+]?\d*\.\d+|\d+)", "technical"),
            (r"the aesthetic quality of video .+ is better than (\d+)% of videos, with normalized score ([-+]?\d*\.\d+|\d+)", "aesthetic")
        ]
        
        current_dataset = None
        
        for line in output.split('\n'):
            for pattern, key in dataset_patterns:
                match = re.search(pattern, line)
                if match:
                    if key == "dataset":
                        current_dataset = match.group(1)
                        result["dataset_comparisons"][current_dataset] = {}
                    elif key == "technical" and current_dataset:
                        result["dataset_comparisons"][current_dataset]["technical_quality"] = {
                            "percentile": int(match.group(1)),
                            "normalized_score": float(match.group(2))
                        }
                    elif key == "aesthetic" and current_dataset:
                        result["dataset_comparisons"][current_dataset]["aesthetic_quality"] = {
                            "percentile": int(match.group(1)),
                            "normalized_score": float(match.group(2))
                        }
    
    return result

def evaluate_video_json(video_path, opt_path="./dover.yml", device="cuda", fusion=False):
    """
    Evaluate one video and return the result as JSON.

    Args:
        video_path (str): path of the video file
        opt_path (str): path of the DOVER config file
        device (str): 'cuda' or 'cpu'
        fusion (bool): whether to report the fused score as well

    Returns:
        str: the evaluation result in JSON format
    """
    result = evaluate_video(video_path, opt_path, device, fusion)
    return json.dumps(result, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # small command line interface, mainly for debugging
    import argparse

    parser = argparse.ArgumentParser(description="DOVER video quality evaluation")
    parser.add_argument("video_path", type=str, help="path of the video file")
    parser.add_argument("-o", "--opt", type=str, default="./dover.yml", help="path of the config file")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="device to run on")
    parser.add_argument("-f", "--fusion", action="store_true", help="also report the fused score")
    parser.add_argument("-j", "--json", action="store_true", help="print the result as JSON")
    
    args = parser.parse_args()
    
    if args.json:
        print(evaluate_video_json(args.video_path, args.opt, args.device, args.fusion))
    else:
        result = evaluate_video(args.video_path, args.opt, args.device, args.fusion)
