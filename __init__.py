from . import glm_image_sdnq_nodes as _sdnq_nodes

try:
    from . import glm_image_sdnq2_nodes as _sdnq2_nodes
except Exception:
    _sdnq2_nodes = None


def _merge_mappings(*modules):
    node_class_mappings = {}
    node_display_name_mappings = {}
    for module in modules:
        if module is None:
            continue
        node_class_mappings.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
        node_display_name_mappings.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    return node_class_mappings, node_display_name_mappings


# Prefer sdnq2 when both define the same nodes.
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = _merge_mappings(
    _sdnq_nodes,
    _sdnq2_nodes,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
