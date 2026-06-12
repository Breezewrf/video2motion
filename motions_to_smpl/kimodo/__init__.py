"""Bundled Kimodo conversion helpers."""

import sys

# The copied Kimodo modules use absolute imports such as ``kimodo.geometry``.
# Register this package under that name when imported via ``motions_to_smpl``.
sys.modules.setdefault("kimodo", sys.modules[__name__])
