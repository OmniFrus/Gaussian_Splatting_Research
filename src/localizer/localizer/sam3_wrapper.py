import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import logging
import time
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3Wrapper:
    def __init__(self, default_prompt: str = "chair", resolution: int = 256):
        self.default_prompt = default_prompt
        self.requested_resolution = resolution
        self.resolution = resolution
        self.model_expected_resolution = 1008
        self.logger = logging.getLogger(__name__)

        self.model = build_sam3_image_model(device="cpu")
        self.model = self.model.float()
        self.model.eval()

        self.processor = self._build_processor(self.resolution)

        self.logger.info(
            "SAM3Wrapper initialized | prompt=%s | processor_resolution=%d | expected_backbone_resolution=%d",
            self.default_prompt,
            self.resolution,
            self.model_expected_resolution,
        )
        if self.resolution != self.model_expected_resolution:
            self.logger.warning(
                "SAM3 processor resolution is %d, but current SAM3 backbone is built around %d. "
                "If you hit RoPE assertion errors, this is likely the cause.",
                self.resolution,
                self.model_expected_resolution,
            )

    def _build_processor(self, resolution: int):
        self.logger.info("Building SAM3 processor with resolution=%d", resolution)
        return Sam3Processor(
            self.model,
            resolution=resolution,
            device="cpu",
        )

    def set_prompt(self, prompt: str):
        self.default_prompt = prompt

    def predict(self, image_bgr: np.ndarray, prompt: str = None):
        if prompt is None:
            prompt = self.default_prompt

        h, w = image_bgr.shape[:2]
        self.logger.info(
            "SAM3 predict start | prompt=%s | input_shape=(%d,%d,%d) | processor_resolution=%d",
            prompt,
            h,
            w,
            image_bgr.shape[2] if image_bgr.ndim == 3 else -1,
            self.resolution,
        )
        image_rgb = image_bgr[:, :, ::-1]
        pil_image = Image.fromarray(image_rgb)

        t0 = time.perf_counter()
        try:
            with torch.autocast(device_type="cpu", enabled=False):
                state = self.processor.set_image(pil_image)
                output = self.processor.set_text_prompt(state=state, prompt=prompt)
        except AssertionError as err:
            # Fallback path: the public SAM3 backbone currently expects 1008-token geometry.
            # If we started with a lower resolution for performance, recover once by rebuilding
            # the processor at 1008 so the ROS node stays alive.
            if self.resolution != self.model_expected_resolution:
                self.logger.warning(
                    "SAM3 assertion at resolution=%d. Retrying once with resolution=%d.",
                    self.resolution,
                    self.model_expected_resolution,
                )
                self.resolution = self.model_expected_resolution
                self.processor = self._build_processor(self.resolution)
                with torch.autocast(device_type="cpu", enabled=False):
                    state = self.processor.set_image(pil_image)
                    output = self.processor.set_text_prompt(state=state, prompt=prompt)
            else:
                msg = (
                    "SAM3 backbone assertion failed during set_image/set_text_prompt. "
                    f"Likely resolution incompatibility: processor_resolution={self.resolution}, "
                    f"expected_backbone_resolution={self.model_expected_resolution}, "
                    f"input_shape=({h},{w},{image_bgr.shape[2] if image_bgr.ndim == 3 else -1}). "
                    "Try setting processor resolution to 1008 or use a SAM3 model variant trained/configured for lower resolution."
                )
                self.logger.exception(msg)
                raise RuntimeError(msg) from err

        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        self.logger.info(
            "SAM3 predict end | elapsed_ms=%d | masks=%d | boxes=%d | scores=%d",
            elapsed_ms,
            0 if masks is None else len(masks),
            0 if boxes is None else len(boxes),
            0 if scores is None else len(scores),
        )
        return masks, boxes, scores

    def best_mask(self, image_bgr: np.ndarray, prompt: str = None):
        masks, boxes, scores = self.predict(image_bgr, prompt)

        if masks is None or len(masks) == 0:
            self.logger.info("SAM3 best_mask: no masks returned")
            return None, None, None

        if scores is None or len(scores) == 0:
            idx = 0
        else:
            idx = int(torch.argmax(scores).item())

        mask = masks[idx].detach().cpu().numpy()
        mask = np.squeeze(mask)
        mask = (mask > 0).astype(np.uint8)

        box = None
        if boxes is not None and len(boxes) > idx:
            box = boxes[idx].detach().cpu().numpy()

        score = None
        if scores is not None and len(scores) > idx:
            score = float(scores[idx].detach().cpu().item())

        self.logger.info(
            "SAM3 best_mask selected | idx=%d | score=%s | mask_shape=%s | box=%s",
            idx,
            "None" if score is None else f"{score:.4f}",
            tuple(mask.shape),
            None if box is None else box.tolist(),
        )
        return mask, box, score