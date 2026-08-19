"""SAM2 outfit segmentation using safetensors weights (safe, no pickle execution).

Requires:
  pip install git+https://github.com/facebookresearch/sam2.git
  pip install safetensors  (usually already installed with ComfyUI)

Model: download sam2_hiera_tiny.safetensors (or similar) from
  https://huggingface.co/Kijai/sam2-safetensors
  and place in ComfyUI/models/sams/ or ComfyUI/models/sam2/
"""

from __future__ import annotations

import glob
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Availability ───────────────────────────────────────────────────────────────

def check_dependencies() -> dict:
    """Return {available, missing} for sam2 + safetensors packages."""
    missing = []
    try:
        import sam2  # noqa: F401
    except ImportError:
        missing.append("sam2")
    try:
        import safetensors  # noqa: F401
    except ImportError:
        missing.append("safetensors")
    return {"available": len(missing) == 0, "missing": missing}


def find_sam2_model(model_dirs: list[str]) -> Optional[str]:
    """Scan ComfyUI model directories for a SAM2 safetensors file."""
    for d in model_dirs:
        hits = glob.glob(os.path.join(d, "sam2*.safetensors"))
        hits += glob.glob(os.path.join(d, "sam2.1*.safetensors"))
        hits = [h for h in hits if os.path.isfile(h)]
        if hits:
            tiny = [h for h in hits if "tiny" in Path(h).stem.lower() or "_t." in Path(h).stem]
            return tiny[0] if tiny else hits[0]
    return None


# ── Config detection ───────────────────────────────────────────────────────────

def _detect_config(stem: str) -> str:
    """Pick the sam2 hydra config YAML name that matches a safetensors filename."""
    s = stem.lower()
    v21 = "2.1" in s or "21" in s.replace(".", "")

    if "large" in s or s.endswith("_l"):
        suffix = "l"
    elif "base" in s or "_b+" in s or "_bplus" in s:
        suffix = "b+"
    elif "small" in s or s.endswith("_s"):
        suffix = "s"
    else:
        suffix = "t"

    if v21:
        return f"configs/sam2.1/sam2.1_hiera_{suffix}.yaml"
    return f"configs/sam2/sam2_hiera_{suffix}.yaml"


# ── Segmenter ─────────────────────────────────────────────────────────────────

class SAM2Segmenter:
    """Lazy-loaded SAM2 image predictor backed by a safetensors checkpoint.

    The predictor is built once and reused across calls. Weights are loaded
    with safetensors.torch.load_file() — no pickle deserialization occurs.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self._model_path = model_path
        self._device = device
        self._predictor = None

    def _load(self) -> None:
        if self._predictor is not None:
            return

        from safetensors.torch import load_file
        import sam2 as _sam2_pkg
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from omegaconf import OmegaConf
        from hydra.utils import instantiate

        config_rel = _detect_config(Path(self._model_path).stem)
        logger.info("SAM2: building architecture from config=%s", config_rel)

        # Load the YAML directly from the sam2 package directory instead of
        # going through Hydra's global compose() — other custom nodes (e.g.
        # Comfyui-SecNodes) call GlobalHydra.clear() and re-initialize Hydra
        # with their own config module, which breaks the standard build_sam2()
        # call when it fires after those nodes have loaded.
        sam2_pkg_dir = Path(_sam2_pkg.__file__).parent
        config_path = sam2_pkg_dir / config_rel
        if not config_path.exists():
            raise FileNotFoundError(f"SAM2 config not found at {config_path}")

        cfg = OmegaConf.load(config_path)
        # Apply the same postprocessing overrides that build_sam2() normally adds.
        OmegaConf.update(cfg, "model.sam_mask_decoder_extra_args", {
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        }, merge=True, force_add=True)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
        model = model.to(self._device)
        model.eval()

        logger.info("SAM2: loading safetensors weights from %s", self._model_path)
        state_dict = load_file(self._model_path, device="cpu")

        # Kijai's conversion stores the flat state dict directly (no nesting).
        # Guard against any "model." prefix that might appear in other conversions.
        if all(k.startswith("model.") for k in list(state_dict.keys())[:5]):
            state_dict = {k[len("model."):]: v for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("SAM2: %d missing keys after load", len(missing))
        if unexpected:
            logger.warning("SAM2: %d unexpected keys after load", len(unexpected))

        self._predictor = SAM2ImagePredictor(model)
        logger.info("SAM2: predictor ready")

    def segment(
        self,
        image_path: str,
        point_x: float,
        point_y: float,
        point_label: int = 1,
        output_path: Optional[str] = None,
    ) -> str:
        """Segment the region near (point_x, point_y) in the image.

        Args:
            image_path: Path to the source image.
            point_x: Normalized x coordinate [0, 1] (left→right).
            point_y: Normalized y coordinate [0, 1] (top→bottom).
            point_label: 1 = foreground (include), 0 = background (exclude).
            output_path: Save path for RGBA PNG. Auto-generated if None.

        Returns:
            Path to the saved RGBA PNG (outfit on transparent background).
        """
        import numpy as np
        import torch
        from PIL import Image

        self._load()

        image_pil = Image.open(image_path).convert("RGB")
        w, h = image_pil.size
        image_np = np.array(image_pil)

        px = float(point_x) * w
        py = float(point_y) * h

        with torch.inference_mode():
            self._predictor.set_image(image_np)
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array([[px, py]], dtype=np.float32),
                point_labels=np.array([point_label], dtype=np.int32),
                multimask_output=True,
            )

        best = int(np.argmax(scores))
        mask = masks[best]  # bool H×W

        result = image_pil.convert("RGBA")
        result.putalpha(Image.fromarray((mask * 255).astype(np.uint8)))

        if output_path is None:
            uid = uuid.uuid4().hex[:12]
            output_path = os.path.join(
                os.path.dirname(os.path.abspath(image_path)),
                f"_outfit_seg_{uid}.png",
            )

        result.save(output_path, "PNG")
        logger.info("SAM2: saved segmented result → %s", output_path)
        return output_path
