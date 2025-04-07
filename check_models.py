from huggingface_hub import snapshot_download
import os

MODEL_PATHS = {
    "RoomDreamingModel": {
        "repo": "NTUHCILAB/RoomDreamingModel",
        "path": "RoomDreamingModel"
    },
    "ControlNetDepth": {
        "repo": "lllyasviel/control_v11f1p_sd15_depth",
        "path": "ControlNetModel/depth"
    },
    "ControlNetSeg": {
        "repo": "lllyasviel/control_v11p_sd15_seg",
        "path": "ControlNetModel/seg"
    },
    "DepthEstimator": {
        "repo": "lllyasviel/Annotators",
        "path": "ImageAnalysisModel/depth"
    },
    "SegmentationEstimator": {
        "repo": "openmmlab/upernet-convnext-small",
        "path": "ImageAnalysisModel/seg"
    }
}


def check_and_download_models(base_path):
    for name, config in MODEL_PATHS.items():
        full_path = os.path.join(base_path, config["path"])
        if not os.path.exists(full_path):
            print(f"Downloading {name} to {full_path}")
            snapshot_download(repo_id=config["repo"],
                              local_dir=full_path,
                              local_dir_use_symlinks=False)
        else:
            print(f"{name} already exists at {full_path}")
