# from models.wan_video_dit import WanModel
from models.wan_video_vae import WanVideoVAE
from customize_model import CustomizeWanModel


model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (state_dict_keys_hash, state_dict_keys_hash_with_shape, model_names, model_classes, model_resource)
    (None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [CustomizeWanModel], "civitai"),
    (None, "aafcfd9672c3a2456dc46e1cb6e52c70", ["wan_video_dit"], [CustomizeWanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [CustomizeWanModel], "civitai"),
    (None, "6d6ccde6845b95ad9114ab993d917893", ["wan_video_dit"], [CustomizeWanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [CustomizeWanModel], "civitai"),
    (None, "349723183fc063b2bfc10bb2835cf677", ["wan_video_dit"], [CustomizeWanModel], "civitai"),
    (None, "efa44cddf936c70abd0ea28b6cbe946c", ["wan_video_dit"], [CustomizeWanModel], "civitai"),
    (None, "3ef3b1f8e1dab83d5b71fd7b617f859f", ["wan_video_dit"], [CustomizeWanModel], "civitai"),
    (None, "70ddad9d3a133785da5ea371aae09504", ["wan_video_dit"], [CustomizeWanModel], "civitai"),
    (None, "26bde73488a92e64cc20b0a7485b9e5b", ["wan_video_dit"], [CustomizeWanModel], "civitai"),
    (None, "ac6a5aa74f4a0aab6f64eb9a72f19901", ["wan_video_dit"], [CustomizeWanModel], "civitai"),
    (None, "b61c605c2adbd23124d152ed28e049ae", ["wan_video_dit"], [CustomizeWanModel], "civitai"),
    (None, "cb104773c6c2cb6df4f9529ad5c60d0b", ["wan_video_dit"], [CustomizeWanModel], "diffusers"),
    (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "civitai"),
]
