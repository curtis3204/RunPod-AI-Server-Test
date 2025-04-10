import base64
import torch
import numpy as np
import cv2
from diffusers import AutoPipelineForText2Image, ControlNetModel
import gc
from PIL import Image
import PIL.Image
from controlnet_aux import LeresDetector
from Image_Segmentor.image_segmentor import ImageSegmentor
import os

MODEL_BASE_PATH = os.getenv("MODEL_STORAGE", "/runpod-volume/models")

# Global pipeline initialization
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Using device: {device}]")

controlnets = [
    ControlNetModel.from_pretrained(
        f"{MODEL_BASE_PATH}/ControlNetModel/depth",
        torch_dtype=torch.float16,
        local_files_only=True,
        use_safetensors=False,
    ),
    ControlNetModel.from_pretrained(
        f"{MODEL_BASE_PATH}/ControlNetModel/seg",
        torch_dtype=torch.float16,
        local_files_only=True,
        use_safetensors=False,
    )
]

try:
    print("[Loading RoomDreaming Pipeline...]")
    image_generation_pipe = AutoPipelineForText2Image.from_pretrained(
        f"{MODEL_BASE_PATH}/RoomDreamingModel",
        controlnet=controlnets,
        torch_dtype=torch.float16,
        local_files_only=True,
        use_safetensors=False,
        safety_checker=None).to(device)
    image_generation_pipe.enable_xformers_memory_efficient_attention()
    print("[RoomDreaming Pipeline loaded successfully]")

    # --------------------- Load depth and segmentation estimators --------------------- #
    print("[Loading depth and segmentation estimators...]")
    # Load depth estimator
    leres = LeresDetector.from_pretrained(
        f"{MODEL_BASE_PATH}/ImageAnalysisModel/depth", )
    # Load segmentation estimator
    segmentor = ImageSegmentor()
    print("[Depth and segmentation estimators loaded successfully]")

except Exception as e:
    print(f"[Failed to load pipeline: {str(e)}]")
    raise


def handler(event):
    try:
        input_data = event.get("input", {})
        print(f"Received input: {input_data}")
        task_type = input_data.get("task_type", "")
        if task_type == "image_generation":
            print("[Start Image Generation]")
            pos_prompt = input_data.get("pos_prompt", "")
            neg_prompt = input_data.get("neg_prompt", "")
            depth_image = input_data.get("depth_image", "")
            seg_image = input_data.get("seg_image", "")
            depth_weight = float(input_data.get("depth_control_weight", 0.5))
            seg_weight = float(input_data.get("seg_control_weight", 0.5))
            num_steps = int(input_data.get("num_steps", 25))
            guidance_scale = float(input_data.get("guidance_scale", 7.5))
            seed = int(input_data.get("seed", 0))

            print(f"pos_prompt: {pos_prompt}")
            print(f"neg_prompt: {neg_prompt}")
            print(
                f"depth_image: {depth_image[:50]}... (length: {len(depth_image)})"
            )
            print(f"seg_image: {seg_image[:50]}... (length: {len(seg_image)})")
            print(f"depth_weight: {depth_weight}")
            print(f"seg_weight: {seg_weight}")
            print(f"num_steps: {num_steps}")
            print(f"guidance_scale: {guidance_scale}")
            print(f"seed: {seed}")

            if not (pos_prompt and depth_image and seg_image):
                return {"error": "Missing required inputs"}

            depth_pil = decode_base64_to_pil(depth_image, "depth_image")
            seg_pil = decode_base64_to_pil(seg_image, "seg_image")

            with torch.no_grad():
                image = image_generation_pipe(
                    prompt=pos_prompt,
                    negative_prompt=neg_prompt,
                    image=[depth_pil, seg_pil],
                    controlnet_conditioning_scale=[depth_weight, seg_weight],
                    num_inference_steps=25,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                ).images[0]

            # Convert PIL Image to base64 with data URI prefix
            image_base64_with_prefix = convert_image_to_base64(image)
            torch.cuda.empty_cache()
            gc.collect()

            return {"image": image_base64_with_prefix}
        elif task_type == "image_analysis":
            print("[Start Image Analysis]")
            original_image = input_data.get("original_image", "")
            if not original_image:
                return {"error": "Missing required inputs"}

            print(
                f"original_image: {original_image[:50]}... (length: {len(original_image)})"
            )

            original_image = decode_base64_to_np(original_image)
            result_dict = image_analysis_pipe(original_image)

            depth_image = convert_image_to_base64(result_dict["depth_image"])
            seg_image = convert_image_to_base64(result_dict["seg_image"])
            ade20k_id_list = result_dict["ade20k_id_list"]

            torch.cuda.empty_cache()
            gc.collect()

            return {
                "depth_image": depth_image,
                "seg_image": seg_image,
                "ade20k_id_list": ade20k_id_list
            }
        else:
            return {"error": "Invalid task type"}

    except Exception as e:
        print(f"Handler error: {str(e)}")
        return {"error": str(e)}


@torch.inference_mode()
def process_depth(original_image: np.ndarray, ) -> PIL.Image.Image:
    print("  [In Processing depth]")

    # original_image = resize_image(HWC3(original_image), image_resolution)
    H, W = original_image.shape[:2]
    min_side = min(H, W)

    depth_image = leres(original_image,
                        detect_resolution=min_side,
                        image_resolution=min_side)

    print("  [Finsih Processing depth]")

    return depth_image


@torch.inference_mode()
def process_segmentation(
    input_image: np.ndarray, ) -> dict[PIL.Image.Image, list]:
    print("  [In Processing segmentation]")
    H, W = input_image.shape[:2]
    max_side = max(H, W)

    seg_image, ade20k_id_list = segmentor(
        image=input_image,
        image_resolution=max_side,
        detect_resolution=max_side,
    )

    print("  [Finsih Processing segmentation]")
    return {"seg_image": seg_image, "ade20k_id_list": ade20k_id_list}


@torch.autocast("cuda")
def image_analysis_pipe(original_image: np.ndarray, ) -> list[PIL.Image.Image]:

    print("[Start Image Analysis]")
    print("[Processing depth]")
    depth_image = process_depth(original_image)
    print("[Processing segmentation]")
    seg_result = process_segmentation(original_image)
    seg_image = seg_result["seg_image"]
    ade20k_id_list = seg_result["ade20k_id_list"]

    result_dict = {}
    result_dict["depth_image"] = depth_image
    result_dict["seg_image"] = seg_image
    result_dict["ade20k_id_list"] = ade20k_id_list
    print("[Finish Image Analysis]")

    return result_dict


# --------------------- helper functions --------------------- #
def convert_image_to_base64(image: PIL.Image.Image) -> str:
    # Convert PIL Image to base64 with data URI prefix
    image_array = np.array(image)
    _, buffer = cv2.imencode(".jpeg",
                             cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    image_base64_with_prefix = f"data:image/jpeg;base64,{image_base64}"
    return image_base64_with_prefix


def decode_base64_to_pil(base64_str, input_name="unknown"):
    try:
        print(
            f"[Decoding {input_name} base64 string: {base64_str[:50]}... (length: {len(base64_str)})]"
        )
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",", 1)[1]
            print(
                f"Stripped data URI prefix from {input_name}. New length: {len(base64_str)}"
            )

        padding_needed = len(base64_str) % 4
        if padding_needed:
            base64_str += "=" * (4 - padding_needed)
            print(
                f"Added {4 - padding_needed} padding characters to {input_name}"
            )

        img_data = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode {input_name} into an image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        print(
            f"[Decoded {input_name} to PIL Image with size: {pil_image.size}]")
        return pil_image
    except Exception as e:
        print(f"Error decoding {input_name}: {str(e)}")
        raise


def decode_base64_to_np(base64_str):
    try:
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",", 1)[1]

        padding_needed = len(base64_str) % 4
        if padding_needed:
            base64_str += "=" * (4 - padding_needed)

        img_data = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image into an array")

        return img
    except Exception as e:
        print(f"Error decoding image: {str(e)}")
        raise


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
