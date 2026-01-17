import gc
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
from PIL import Image

from comfy_api.latest import io


# -------------------------
# Internal cache utilities
# -------------------------

@dataclass(frozen=True)
class _PipeKey:
    model_id: str
    device: str
    dtype: str


_PIPE_CACHE: Dict[_PipeKey, object] = {}


def _torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def _get_device(device_name: str) -> torch.device:
    # ComfyUI typically runs on CUDA; we still support CPU for completeness.
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_pipeline(model_id: str, device_name: str, dtype_name: str):
    """Load and cache GlmImagePipeline."""
    key = _PipeKey(model_id=model_id, device=device_name, dtype=dtype_name)
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]

    # Import here to avoid slowing ComfyUI startup if the user doesn't use the nodes.
    # GLM-Image pipeline exists only in fairly recent diffusers builds.
    # Newer diffusers sometimes re-export pipelines at top-level.
    try:
        from diffusers import GlmImagePipeline  # type: ignore
    except Exception:
        try:
            from diffusers.pipelines.glm_image import GlmImagePipeline  # type: ignore
        except Exception as e:
            raise ModuleNotFoundError(
                "GLM-Image pipeline not found in your diffusers install. "
                "Install diffusers/transformers from source (git main) and restart ComfyUI."
            ) from e

    torch_dtype = _torch_dtype(dtype_name)

    # The HF docs show device_map="cuda" usage. In many Comfy installs, accelerate is present.
    # We'll prefer device_map when CUDA requested; fallback to .to(device).
    pipe = GlmImagePipeline.from_pretrained(model_id, torch_dtype=torch_dtype)

    # Prefer device_map if available (accelerate), but fall back to a plain .to()
    if device_name == "cuda" and torch.cuda.is_available():
        try:
            pipe = GlmImagePipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="cuda",
            )
        except Exception:
            pipe.to(torch.device("cuda"))
    else:
        pipe.to(_get_device(device_name))

    _PIPE_CACHE[key] = pipe
    return pipe


def _maybe_clear_cuda_cache():
    # Best-effort cleanup when users chain many generations.
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# -------------------------
# Image conversion helpers
# -------------------------

def _comfy_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI IMAGE (float 0..1, HxWxC or 1xHxWxC) to PIL RGB."""
    t = image_tensor
    if t is None:
        raise ValueError("init_image is None")

    # Common formats in Comfy: [H,W,C] or [B,H,W,C]
    if t.dim() == 4:
        t = t[0]
    if t.dim() != 3 or t.shape[-1] not in (3, 4):
        raise ValueError(f"Unexpected image tensor shape: {tuple(t.shape)}")

    t = t.detach().to("cpu").clamp(0, 1)
    if t.shape[-1] == 4:
        t = t[..., :3]

    arr = (t * 255.0).round().to(torch.uint8).numpy()
    return Image.fromarray(arr, mode="RGB")


def _pil_to_comfy_image(img: Image.Image) -> torch.Tensor:
    """Convert PIL RGB to ComfyUI IMAGE tensor float32 0..1 in [H,W,C]."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = torch.from_numpy(__import__("numpy").array(img)).to(torch.float32) / 255.0
    return arr


# -------------------------
# Nodes
# -------------------------

class GLMImageGenerate(io.ComfyNode):
    """Text-to-image generation using GLM-Image (diffusers GlmImagePipeline)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GLMImageGenerate",
            display_name="GLM-Image Generate (T2I)",
            category="GLM-Image",
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="A poster with the title \"HELLO\" in large text.",
                ),
                io.Int.Input("width", default=1024, min=512, max=2048, step=32, display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=1024, min=512, max=2048, step=32, display_mode=io.NumberDisplay.number),
                io.Int.Input("steps", default=30, min=1, max=100, step=1, display_mode=io.NumberDisplay.number),
                io.Float.Input("guidance_scale", default=1.5, min=0.0, max=20.0, step=0.1, display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=42, min=-1, max=2**31 - 1, step=1, display_mode=io.NumberDisplay.number),
                io.String.Input("model_id", default="zai-org/GLM-Image", multiline=False),
                io.Combo.Input("device", options=["cuda", "cpu"]),
                io.Combo.Input("dtype", options=["bfloat16", "float16", "float32"]),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        model_id: str,
        device: str,
        dtype: str,
    ) -> io.NodeOutput:
        # Validate for GLM-Image requirements: multiples of 32.
        if width % 32 != 0 or height % 32 != 0:
            raise ValueError("width and height must be multiples of 32")

        pipe = _load_pipeline(model_id=model_id, device_name=device, dtype_name=dtype)

        if seed is None or seed < 0:
            seed = torch.seed() % (2**31 - 1)

        gen_device = _get_device(device)
        generator = torch.Generator(device=gen_device)
        generator.manual_seed(int(seed))

        # Run
        out = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        )

        img = out.images[0]  # PIL
        comfy_img = _pil_to_comfy_image(img)

        _maybe_clear_cuda_cache()
        return io.NodeOutput(comfy_img)


class GLMImageImageToImage(io.ComfyNode):
    """Image-to-image generation (edit/transform) using GLM-Image pipeline."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GLMImageImageToImage",
            display_name="GLM-Image Generate (I2I)",
            category="GLM-Image",
            inputs=[
                io.Image.Input("image"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="Replace the background with a subway station.",
                ),
                io.Int.Input("width", default=1024, min=512, max=2048, step=32, display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=1024, min=512, max=2048, step=32, display_mode=io.NumberDisplay.number),
                io.Int.Input("steps", default=30, min=1, max=100, step=1, display_mode=io.NumberDisplay.number),
                io.Float.Input("guidance_scale", default=1.5, min=0.0, max=20.0, step=0.1, display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=42, min=-1, max=2**31 - 1, step=1, display_mode=io.NumberDisplay.number),
                io.String.Input("model_id", default="zai-org/GLM-Image", multiline=False),
                io.Combo.Input("device", options=["cuda", "cpu"]),
                io.Combo.Input("dtype", options=["bfloat16", "float16", "float32"]),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        model_id: str,
        device: str,
        dtype: str,
    ) -> io.NodeOutput:
        if width % 32 != 0 or height % 32 != 0:
            raise ValueError("width and height must be multiples of 32")

        pipe = _load_pipeline(model_id=model_id, device_name=device, dtype_name=dtype)

        if seed is None or seed < 0:
            seed = torch.seed() % (2**31 - 1)

        gen_device = _get_device(device)
        generator = torch.Generator(device=gen_device)
        generator.manual_seed(int(seed))

        pil = _comfy_image_to_pil(image)

        out = pipe(
            prompt=prompt,
            image=[pil],
            width=width,
            height=height,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        )

        img = out.images[0]
        comfy_img = _pil_to_comfy_image(img)

        _maybe_clear_cuda_cache()
        return io.NodeOutput(comfy_img)
