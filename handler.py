import base64
import torch
import numpy as np
import cv2
from diffusers import AutoPipelineForText2Image, ControlNetModel
import gc

# Global pipeline initialization (loaded once at container startup)
device = "cuda" if torch.cuda.is_available() else "cpu"
controlnets = [
    ControlNetModel.from_pretrained(
        "/data/ControlNetModel/depth",
        torch_dtype=torch.float16,
        local_files_only=True
    ),
    ControlNetModel.from_pretrained(
        "/data/ControlNetModel/seg",
        torch_dtype=torch.float16,
        local_files_only=True
    )
]
pipe = AutoPipelineForText2Image.from_pretrained(
    "/data/RoomDreamingModel",
    controlnet=controlnets,
    torch_dtype=torch.float16,
    local_files_only=True
).to(device)
pipe.enable_xformers_memory_efficient_attention()  # Optimize memory usage

def decode_base64_image(base64_str):
    """Convert base64 string to tensor."""
    img_data = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    return img_tensor

def handler(event):
    """RunPod serverless handler."""
    try:
        # Parse input
        input_data = event.get("input", {})
        pos_prompt = input_data.get("pos_prompt", "")
        neg_prompt = input_data.get("neg_prompt", "")
        depth_image = input_data.get("depth_image", "")  # Base64 encoded
        seg_image = input_data.get("seg_image", "")      # Base64 encoded
        depth_weight = float(input_data.get("depth_control_weight", 0.5))
        seg_weight = float(input_data.get("seg_control_weight", 0.5))

        if not (pos_prompt and depth_image and seg_image):
            return {"error": "Missing required inputs"}

        # Decode control images
        depth_tensor = decode_base64_image(depth_image)
        seg_tensor = decode_base64_image(seg_image)

        # Generate image with dual ControlNet
        with torch.no_grad():
            image = pipe(
                prompt=pos_prompt,
                negative_prompt=neg_prompt,
                control_image=[depth_tensor, seg_tensor],
                controlnet_conditioning_scale=[depth_weight, seg_weight],
                guidance_scale=7.5,
                num_inference_steps=25
            ).images[0]

        # Convert to base64
        _, buffer = cv2.imencode(".png", np.array(image))
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        # Clean up
        torch.cuda.empty_cache()
        gc.collect()

        return {"image": image_base64}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})