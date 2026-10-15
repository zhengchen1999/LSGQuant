import imageio
import torch
from PIL import Image
from accelerate import Accelerator
from pathlib import Path
from safetensors.torch import load_file
from torchvision import transforms
from datasets.degradation import (RandomBlur, RandomResize, RandomNoise, RandomJPEGCompression, RandomVideoCompression, DegradationsWithShuffle)
from datasets.utils import (
    load_videos_with_root,
    read_video_frames,
    random_crop_frames,
    paired_random_crop_video
)
import random
import numpy as np
import yaml
import math
import os


class RealSRValDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            lq_root: str,
            hq_root: str,
            prompt_cache: str = "prompt_embeddings",
    ):
        super().__init__()

        self.lq_videos = os.listdir(lq_root)
        if not os.path.exists(hq_root):
            self.hq_videos = []
        else:
            self.hq_videos = os.listdir(hq_root)
        self.lq_root = lq_root
        self.hq_root = hq_root
        empty_prompt_dir = os.path.join(prompt_cache, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors")
        self.empty_prompt = load_file(empty_prompt_dir)["prompt_emb"]

    def __len__(self) -> int:
        return len(self.lq_videos)

    def __getitem__(self, index: int):
        if isinstance(index, list):
            return index

        lq_video = os.path.join(self.lq_root, self.lq_videos[index])
        # Current shape of frames: [F, C, H, W]
        lq_frames = self.preprocess(lq_video)

        if not os.path.exists(self.hq_root):
            hq_frames = None
        else:
            hq_video = os.path.join(self.hq_root, self.hq_videos[index])
            hq_frames = self.preprocess(hq_video, crop=True)
        return {
            "name": self.lq_videos[index],
            "hq_video": [hq_frames],
            "lq_video": [lq_frames],
            "prompt_embed": self.empty_prompt.unsqueeze(0),
        }

    def preprocess(self, video_path: Path, crop=False):
        """
        Loads and preprocesses a video.

        Args:
            video_path: Path to the video file to load.

        Returns:
            torch.Tensor: Video tensor of shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width
        """
        def crop_bottomright_to_16(image: Image.Image) -> Image.Image:
            w, h = image.size
            w_ = w // 16 * 16
            h_ = h // 16 * 16
            return image.crop((0, 0, w_, h_))  # crop from the top-left corner
        frame_list = read_video_frames(video_path)
        frame_list = [Image.fromarray(f) for f in frame_list]
        if crop:
            # 1. make height and width multiples of 16
            frame_list = [crop_bottomright_to_16(f) for f in frame_list]
            # 2. pad to 8n+1 frames
            target_len = ((max(len(frame_list) - 1, 0) + 7) // 8) * 8 + 1
            if len(frame_list) < target_len:
                last = frame_list[-1]
                frame_list += [last] * (target_len - len(frame_list))
        return frame_list


class RealSRDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_root: str,
            max_num_frames: int,
            height: int,
            width: int,
            video_column: str = None,
            degradation_config = "diffsynth/data/degradation.yaml",
            prompt_cache: str = "prompt_embeddings",
            nums: int = -1
    ):
        super().__init__()

        data_root = Path(data_root)
        # the prompt cache is keyed by the order of the videos
        if video_column is None:
            self.videos = []
            for root, _, files in os.walk(data_root):
                for f in files:
                    if f.endswith(('.mp4', '.mkv')):
                        self.videos.append(Path(os.path.join(root, f)))
        else:
            self.videos = load_videos_with_root(video_column, data_root)
        empty_prompt_dir = os.path.join(prompt_cache, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors")
        self.empty_prompt = load_file(empty_prompt_dir)["prompt_emb"]
        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transform = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]  # -1, 1
        )

        with open(degradation_config, 'r') as f:
            self.opt = yaml.safe_load(f)

        # Initialize degradation operations
        self.init_degradtion(self.opt)
        self.inter_frames = self.max_num_frames + 10
        self.inter_height = math.ceil((self.height * 1.5) / 16) * 16
        self.inter_width = math.ceil((self.width * 1.5) / 16) * 16
        self.target_h = int(self.height / 4)
        self.target_w = int(self.width / 4)

        # Check if all video files exist
        if any(not path.is_file() for path in self.videos):
            raise ValueError(
                f"Some video files were not found. Please ensure that all video files exist in the dataset directory. Missing file: {next(path for path in self.videos if not path.is_file())}"
            )

        if nums != -1:
            assert nums <= len(self.videos), "too many videos for choosing!"

        self.videos = random.sample(self.videos, nums)

        print(f"Found {len(self.videos)} video files")

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int):
        if isinstance(index, list):
            return index

        video = self.videos[index]
        # Current shape of frames: [F, C, H, W]
        hq_frames, lq_frames = self.preprocess(video)
        prompt_embed = self.empty_prompt
        return {
            "hq_video": hq_frames,
            "lq_video": lq_frames,
            "prompt_embed": prompt_embed,
        }

    def collate_fn(self, samples):
        ret = {"hq_video": [], "lq_video": [], "prompt": [],
               "prompt_embed": []}

        for sample in samples:
            ret["hq_video"].append(sample["hq_video"])
            ret["lq_video"].append(sample["lq_video"])
            ret["prompt_embed"].append(sample["prompt_embed"])

        ret["prompt_embed"] = torch.stack(ret["prompt_embed"], dim=0)
        return ret

    def preprocess(self, video_path: Path):
        """
        Loads and preprocesses a video.

        Args:
            video_path: Path to the video file to load.

        Returns:
            torch.Tensor: Video tensor of shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width
        """
        frame_list = read_video_frames(video_path, self.inter_frames)
        crop_frame_list = random_crop_frames(frame_list, self.inter_frames, self.inter_height, self.inter_width)
        inter_H, interW, _ = crop_frame_list[0].shape
        inter_target_h = int(inter_H / 4)
        inter_target_w = int(interW / 4)

        if isinstance(self.degradation_with_shuffle.degradations[0], list):
            self.degradation_with_shuffle.degradations[0][0].params['target_size'] = (inter_target_h,
                                                                                      inter_target_w)
        else:
            self.degradation_with_shuffle.degradations[1][0].params['target_size'] = (inter_target_h,
                                                                                      inter_target_w)

        input_dict = dict(lqs=crop_frame_list)
        deg_frame_list = self.degrade(input_dict)['lqs']

        hq_frame_list, lq_frame_list = paired_random_crop_video(crop_frame_list, deg_frame_list,
                                                                self.max_num_frames, self.target_h, self.target_w,
                                                                4, video_path)
        hq_frame_list = [Image.fromarray(f) for f in hq_frame_list]
        lq_frame_list = [Image.fromarray(f.astype(np.uint8)) for f in lq_frame_list]

        return hq_frame_list, lq_frame_list

    def init_degradtion(self, opt) -> None:
        # Initialize first degradation operations
        self.random_blur_1 = RandomBlur(
            params=opt['degradation_1']['random_blur']['params'],
            keys=opt['degradation_1']['random_blur']['keys']
        )
        self.random_resize_1 = RandomResize(
            params=opt['degradation_1']['random_resize']['params'],
            keys=opt['degradation_1']['random_resize']['keys']
        )
        self.random_noise_1 = RandomNoise(
            params=opt['degradation_1']['random_noise']['params'],
            keys=opt['degradation_1']['random_noise']['keys']
        )
        self.random_jpeg_1 = RandomJPEGCompression(
            params=opt['degradation_1']['random_jpeg']['params'],
            keys=opt['degradation_1']['random_jpeg']['keys']
        )
        self.random_mpeg_1 = RandomVideoCompression(
            params=opt['degradation_1']['random_mpeg']['params'],
            keys=opt['degradation_1']['random_mpeg']['keys']
        )

        # Initialize second degradation operations
        self.random_blur_2 = RandomBlur(
            params=opt['degradation_2']['random_blur']['params'],
            keys=opt['degradation_2']['random_blur']['keys']
        )
        self.random_resize_2 = RandomResize(
            params=opt['degradation_2']['random_resize']['params'],
            keys=opt['degradation_2']['random_resize']['keys']
        )
        self.random_noise_2 = RandomNoise(
            params=opt['degradation_2']['random_noise']['params'],
            keys=opt['degradation_2']['random_noise']['keys']
        )
        self.random_jpeg_2 = RandomJPEGCompression(
            params=opt['degradation_2']['random_jpeg']['params'],
            keys=opt['degradation_2']['random_jpeg']['keys']
        )
        self.degradation_with_shuffle = DegradationsWithShuffle(
            degradations=opt['degradation_2']['degradation_with_shuffle']['degradations'],
            keys=opt['degradation_2']['degradation_with_shuffle']['keys']
        )

        # Define degradation sequence
        self.first_stage = [
            self.random_blur_1,
            self.random_resize_1,
            self.random_noise_1,
            self.random_jpeg_1,
            self.random_mpeg_1
        ]

        self.second_stage = [
            self.random_blur_2,
            self.random_resize_2,
            self.random_noise_2,
            self.random_jpeg_2,
            self.degradation_with_shuffle
        ]

    def degrade(self, data):
        """
        Apply degradation pipeline to input data

        Args:
            data: dict containing frames to be processed (e.g., {'lqs': frame_list})

        Returns:
            dict: processed data with degraded frames
        """
        # Apply first stage degradations
        for degradation in self.first_stage:
            data = degradation(data)

        # Apply second stage degradations
        for degradation in self.second_stage:
            data = degradation(data)

        return data
