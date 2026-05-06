import os
import logging
import time
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class SAM3Wrapper:
    def __init__(self, default_prompt: str = "", resolution: int = 1008):
        self.default_prompt = default_prompt
        self.requested_resolution = resolution
        self.resolution = resolution
        self.model_expected_resolution = 1008
        self.logger = logging.getLogger(__name__)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_sam3_image_model(device=self.device.type)
        self.model = self.model.to(self.device)
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
                "SAM3 processor resolution is %d, but current SAM3 backbone is built around %d. ",
                self.resolution,
                self.model_expected_resolution,
            )
        print("DEVICE:", self.device)

    def _build_processor(self, resolution: int):
        self.logger.info("Building SAM3 processor with resolution=%d", resolution)
        return Sam3Processor(
            self.model,
            resolution=resolution,
            device=self.model.device.type,
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
            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
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
                with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
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

    def all_masks(self, image_bgr: np.ndarray, prompt: str = None, score_threshold: float = 0.0):
        masks, boxes, scores = self.predict(image_bgr, prompt)

        if masks is None or len(masks) == 0:
            self.logger.info("SAM3 all_masks: no masks returned")
            return [], [], []

        out_masks = []
        out_boxes = []
        out_scores = []

        for i in range(len(masks)):
            score = None
            if scores is not None and len(scores) > i:
                score = float(scores[i].detach().cpu().item())

            if score is not None and score < score_threshold:
                continue

            mask = masks[i].detach().cpu().numpy()
            mask = np.squeeze(mask)
            mask = (mask > 0).astype(np.uint8)

            box = None
            if boxes is not None and len(boxes) > i:
                box = boxes[i].detach().cpu().numpy()

            out_masks.append(mask)
            out_boxes.append(box)
            out_scores.append(score)

        self.logger.info(
            "SAM3 all_masks selected | count=%d | scores=%s",
            len(out_masks),
            out_scores,
        )
        return out_masks, out_boxes, out_scores

    def all_masks_from_state(self, state, prompt: str, score_threshold: float = 0.0):
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type == "cuda",
            dtype=torch.float16
        ):
            output = self.processor.set_text_prompt(state=state, prompt=prompt)

        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        if masks is None or len(masks) == 0:
            return [], [], []

        out_masks = []
        out_boxes = []
        out_scores = []

        for i in range(len(masks)):
            score = None
            if scores is not None and len(scores) > i:
                score = float(scores[i].detach().cpu().item())

            if score is not None and score < score_threshold:
                continue

            mask = masks[i].detach().cpu().numpy()
            mask = np.squeeze(mask)
            mask = (mask > 0).astype(np.uint8)

            box = None
            if boxes is not None and len(boxes) > i:
                box = boxes[i].detach().cpu().numpy()

            out_masks.append(mask)
            out_boxes.append(box)
            out_scores.append(score)

        return out_masks, out_boxes, out_scores
    
    def set_image_once(self, image_bgr: np.ndarray):
        image_rgb = image_bgr[:, :, ::-1]
        pil_image = Image.fromarray(image_rgb)

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type == "cuda",
            dtype=torch.float16
        ):
            state = self.processor.set_image(pil_image)

        return state