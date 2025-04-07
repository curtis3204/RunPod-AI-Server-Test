import sys
from huggingface_hub import snapshot_download

try:
    snapshot_download(repo_id='NTUHCILAB/RoomDreamingModel', local_dir='/data/RoomDreamingModel')
    print("RoomDreamingModel downloaded successfully.")
    snapshot_download(repo_id='lllyasviel/control_v11f1p_sd15_depth', local_dir='/data/ControlNetModel/depth')
    print("ControlNetModel depth downloaded successfully.")
    snapshot_download(repo_id='lllyasviel/control_v11p_sd15_seg', local_dir='/data/ControlNetModel/seg')
    print("ControlNetModel seg downloaded successfully.")
    snapshot_download(repo_id='lllyasviel/Annotators', local_dir='/data/ImageAnalysisModel/depth')
    print("Depth Estimator Model downloaded successfully.")
    snapshot_download(repo_id='openmmlab/upernet-convnext-small', local_dir='/data/ImageAnalysisModel/seg')
    print("Segmentaion Estimator Model downloaded successfully.")

except Exception as e:
    print(f'Error downloading models: {str(e)}')
    sys.exit(1)