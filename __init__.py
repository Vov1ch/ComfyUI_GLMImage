"""ComfyUI custom nodes for Z.ai GLM-Image via Hugging Face diffusers.

This extension follows the new comfy_api.latest extension style (ComfyExtension).

Nodes:
- GLMImageGenerate (Text->Image)
- GLMImageImageToImage (Image->Image)

Notes:
- GLM-Image is heavy (AR 9B + DiT decoder). Expect high VRAM/RAM usage.
- Requires recent diffusers/transformers.
"""

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes import GLMImageGenerate, GLMImageImageToImage


class GLMImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            GLMImageGenerate,
            GLMImageImageToImage,
        ]


async def comfy_entrypoint() -> GLMImageExtension:
    # ComfyUI calls this to load your extension and its nodes.
    return GLMImageExtension()
