import torch
import numpy as np
import diffusers

# IMPORTANT: importing SDNQConfig registers SDNQ into diffusers/transformers
from sdnq import SDNQConfig  # noqa: F401
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model


def pil_to_comfy_image(pil_img):
    """Convert PIL image to ComfyUI IMAGE tensor: float32 [B,H,W,C] in [0..1]."""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    t = torch.from_numpy(arr)[None, ...]  # [1,H,W,C]
    return t


def pick_dtype(dtype_name: str):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def pick_device(device_name: str):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


# -------- pipeline cache --------
_PIPE_CACHE = {}  # key -> pipe


def get_or_load_pipe(
    model_id: str,
    dtype_name: str,
    device_name: str,
    cpu_offload: bool,
    quantized_matmul: bool,
):
    key = (model_id, dtype_name, device_name, cpu_offload, quantized_matmul)
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]

    dtype = pick_dtype(dtype_name)
    device = pick_device(device_name)

    pipe = diffusers.GlmImagePipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    # SDNQ: INT8 matmul (if Triton is available and running on CUDA/XPU)
    if (
        quantized_matmul
        and triton_is_available
        and (
            torch.cuda.is_available()
            or (hasattr(torch, "xpu") and torch.xpu.is_available())
        )
    ):
        pipe.transformer = apply_sdnq_options_to_model(
            pipe.transformer,
            use_quantized_matmul=True,
        )
        # Optional speed-up (may increase compile time / memory):
        # pipe.transformer = torch.compile(pipe.transformer)

    # Placement
    if cpu_offload:
        # accelerate handles transfers
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    # Slight memory/perf tweak
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    _PIPE_CACHE[key] = pipe
    return pipe


class GLMImageSDNQ_LoadPipe:
    """Loads and caches a diffusers.GlmImagePipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": (
                    "STRING",
                    {"default": "Disty0/GLM-Image-SDNQ-4bit-dynamic"},
                ),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "device": (["cuda", "xpu", "cpu"], {"default": "cuda"}),
                "cpu_offload": ("BOOLEAN", {"default": True}),
                "quantized_matmul": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("GLM_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"
    CATEGORY = "GLM-Image-SDNQ"

    def load(self, model_id, dtype, device, cpu_offload, quantized_matmul):
        pipe = get_or_load_pipe(
            model_id=model_id,
            dtype_name=dtype,
            device_name=device,
            cpu_offload=cpu_offload,
            quantized_matmul=quantized_matmul,
        )
        return (pipe,)


class GLMImageSDNQ_Generate:
    """Generates an image with the loaded GLM pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("GLM_PIPE",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "A cute cat"},
                ),
                "width": (
                    "INT",
                    {"default": 1152, "min": 64, "max": 4096, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 4096, "step": 32},
                ),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 1.5, "min": 0.0, "max": 30.0, "step": 0.1},
                ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "GLM-Image-SDNQ"

    @torch.inference_mode()
    def generate(self, pipe, prompt, width, height, steps, guidance_scale, seed):
        gen_device = "cpu"
        if torch.cuda.is_available():
            gen_device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            gen_device = "xpu"

        generator = torch.Generator(device=gen_device).manual_seed(int(seed))

        out = pipe(
            prompt=prompt,
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        )

        pil_img = out.images[0]
        comfy_img = pil_to_comfy_image(pil_img)
        return (comfy_img,)



class GLMImageSDNQ_ImageToImage:
    """
    Image-to-Image: редактирование/перерисовка входного изображения по промпту.
    Принимает IMAGE из ComfyUI, конвертирует в PIL и вызывает pipe(..., image=[pil]).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("GLM_PIPE",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Replace the background with ..."}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "GLM-Image-SDNQ"

    @torch.inference_mode()
    def edit(self, pipe, image, prompt, width, height, steps, guidance_scale, seed):
        # IMAGE в ComfyUI: [B,H,W,C] float32 0..1
        img0 = image[0].detach().cpu().numpy()
        img0 = (np.clip(img0, 0.0, 1.0) * 255.0).astype(np.uint8)

        if img0.shape[-1] == 4:
            img0 = img0[..., :3]

        from PIL import Image as PILImage
        pil = PILImage.fromarray(img0, mode="RGB")

        gen_device = "cpu"
        if torch.cuda.is_available():
            gen_device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            gen_device = "xpu"

        generator = torch.Generator(device=gen_device).manual_seed(int(seed))

        out = pipe(
            prompt=prompt,
            image=[pil],  # можно несколько: [pil, pil2]
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        )
        pil_img = out.images[0]
        return (pil_to_comfy_image(pil_img),)


class GLMImageSDNQ_MultiImageToImage:
    """Image-to-Image with two input images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("GLM_PIPE",),
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Replace the background with ..."}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "GLM-Image-SDNQ"

    @torch.inference_mode()
    def edit(self, pipe, image_a, image_b, prompt, width, height, steps, guidance_scale, seed):
        def to_pil(img_tensor):
            img0 = img_tensor[0].detach().cpu().numpy()
            img0 = (np.clip(img0, 0.0, 1.0) * 255.0).astype(np.uint8)
            if img0.shape[-1] == 4:
                img0 = img0[..., :3]
            from PIL import Image as PILImage

            return PILImage.fromarray(img0, mode="RGB")

        pil_a = to_pil(image_a)
        pil_b = to_pil(image_b)

        gen_device = "cpu"
        if torch.cuda.is_available():
            gen_device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            gen_device = "xpu"

        generator = torch.Generator(device=gen_device).manual_seed(int(seed))

        out = pipe(
            prompt=prompt,
            image=[pil_a, pil_b],
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        )
        pil_img = out.images[0]
        return (pil_to_comfy_image(pil_img),)


class GLMImageSDNQ_FlexibleInput:
    """Generate with 0-2 input images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("GLM_PIPE",),
                "prompt": ("STRING", {"multiline": True, "default": "Replace the background with ..."}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "GLM-Image-SDNQ"

    @torch.inference_mode()
    def run(self, pipe, prompt, width, height, steps, guidance_scale, seed, image_a=None, image_b=None):
        def to_pil(img_tensor):
            img0 = img_tensor[0].detach().cpu().numpy()
            img0 = (np.clip(img0, 0.0, 1.0) * 255.0).astype(np.uint8)
            if img0.shape[-1] == 4:
                img0 = img0[..., :3]
            from PIL import Image as PILImage

            return PILImage.fromarray(img0, mode="RGB")

        images = []
        if image_a is not None:
            images.append(to_pil(image_a))
        if image_b is not None:
            images.append(to_pil(image_b))

        gen_device = "cpu"
        if torch.cuda.is_available():
            gen_device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            gen_device = "xpu"

        generator = torch.Generator(device=gen_device).manual_seed(int(seed))

        kwargs = {
            "prompt": prompt,
            "height": int(height),
            "width": int(width),
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance_scale),
            "generator": generator,
        }
        if images:
            kwargs["image"] = images

        out = pipe(**kwargs)
        pil_img = out.images[0]
        return (pil_to_comfy_image(pil_img),)


NODE_CLASS_MAPPINGS = {
    "GLMImageSDNQ_ImageToImage": GLMImageSDNQ_ImageToImage,
    "GLMImageSDNQ_MultiImageToImage": GLMImageSDNQ_MultiImageToImage,
    "GLMImageSDNQ_FlexibleInput": GLMImageSDNQ_FlexibleInput,
    "GLMImageSDNQ_LoadPipe": GLMImageSDNQ_LoadPipe,
    "GLMImageSDNQ_Generate": GLMImageSDNQ_Generate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GLMImageSDNQ_ImageToImage": "GLM-Image SDNQ 4bit: Image to Image",
    "GLMImageSDNQ_MultiImageToImage": "GLM-Image SDNQ 4bit: Multi Image to Image",
    "GLMImageSDNQ_FlexibleInput": "GLM-Image SDNQ 4bit: Flexible (0-2 Images)",
    "GLMImageSDNQ_LoadPipe": "GLM-Image SDNQ 4bit: Load Pipe",
    "GLMImageSDNQ_Generate": "GLM-Image SDNQ 4bit: Generate",
}
