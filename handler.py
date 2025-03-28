import base64
import torch
import numpy as np
import cv2
from diffusers import AutoPipelineForText2Image, ControlNetModel
import gc
import os

#  Debug: List model directory contents at runtime
# print("[Contents of /data/RoomDreamingModel]")
# for root, dirs, files in os.walk("/data/RoomDreamingModel"):
#     for file in files:
#         print(os.path.join(root, file))

# Global pipeline initialization
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Using device: {device}]")

controlnets = [
    ControlNetModel.from_pretrained(
        "/data/ControlNetModel/depth",
        torch_dtype=torch.float16,
        local_files_only=True,
        use_safetensors=False,
    ),
    ControlNetModel.from_pretrained(
        "/data/ControlNetModel/seg",
        torch_dtype=torch.float16,
        local_files_only=True,
        use_safetensors=False
    )
]
try:
    print("[Loading RoomDreaming Pipeline...]")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "/data/RoomDreamingModel",
        controlnet=controlnets,
        torch_dtype=torch.float16,
        local_files_only=True,
        use_safetensors=False,
        safety_checker=None
    ).to(device)
    pipe.enable_xformers_memory_efficient_attention()
    print("[RoomDreaming Pipeline loaded successfully]")
except Exception as e:
    print(f"[Failed to load pipeline: {str(e)}]")
    raise

def decode_base64_image(base64_str, input_name="unknown"):
    try:
        # Log the raw input for debugging
        print(f"[Decoding {input_name} base64 string: {base64_str[:50]}... (length: {len(base64_str)})]")

        # Strip data URI prefix if present (e.g., "data:image/png;base64,")
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",", 1)[1]
            print(f"Stripped data URI prefix from {input_name}. New length: {len(base64_str)}")
        
        # Add padding if necessary
        padding_needed = len(base64_str) % 4
        if padding_needed:
            base64_str += "=" * (4 - padding_needed)
            print(f"Added {4 - padding_needed} padding characters to {input_name}")

        img_data = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode {input_name} into an image")
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        return img_tensor
    except Exception as e:
        print(f"Error decoding {input_name}: {str(e)}")
        raise

def handler(event):
    try:
        input_data = event.get("input", {})
        print(f"Received input: {input_data}")
        pos_prompt = input_data.get("pos_prompt", "")
        neg_prompt = input_data.get("neg_prompt", "")
        depth_image = input_data.get("depth_image", "")
        seg_image = input_data.get("seg_image", "")
        depth_weight = float(input_data.get("depth_control_weight", 0.5))
        seg_weight = float(input_data.get("seg_control_weight", 0.5))

        if not (pos_prompt and depth_image and seg_image):
            return {"error": "Missing required inputs"}

        depth_tensor = decode_base64_image(depth_image, "depth_image")
        seg_tensor = decode_base64_image(seg_image, "seg_image")

        with torch.no_grad():
            image = pipe(
                prompt=pos_prompt,
                negative_prompt=neg_prompt,
                control_image=[depth_tensor, seg_tensor],
                controlnet_conditioning_scale=[depth_weight, seg_weight],
                guidance_scale=7.5,
                num_inference_steps=25
            ).images[0]

        _, buffer = cv2.imencode(".png", np.array(image))
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        torch.cuda.empty_cache()
        gc.collect()

        return {"image": image_base64}

    except Exception as e:
        print(f"Handler error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})