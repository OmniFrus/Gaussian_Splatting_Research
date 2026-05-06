import time
import cv2
import numpy as np


class SemanticProcessor:
    def __init__(self, sam3, class_registry, timing_logger, logger):
        self.sam3 = sam3
        self.class_registry = class_registry
        self.timing_logger = timing_logger
        self.logger = logger

    def get_prompts(self, current_prompt):
        if current_prompt.strip() == "":
            prompts = self.class_registry.default_classes
            self.logger.info(
                f"Running SAM3 full-scene mode with classes: {prompts}"
            )
        else:
            prompts = [
                p.strip()
                for p in current_prompt.split(",")
                if p.strip()
            ]
            self.logger.info(
                f"Running SAM3 single-class mode with prompt: {current_prompt}"
            )

        return prompts

    def run(self, color_img, depth_img, detection_color_img, current_prompt, frame_count, confidence):
        prompts = self.get_prompts(current_prompt)

        frame_start_time = time.perf_counter()

        semantic_mask = None
        semantic_overlay = np.zeros_like(detection_color_img, dtype=np.uint8)
        semantic_map = np.zeros(depth_img.shape[:2], dtype=np.uint8)
        semantic_map_color = np.zeros_like(detection_color_img, dtype=np.uint8)

        all_boxes = []
        all_scores = []
        all_labels = []

        embed_start = time.perf_counter()
        image_state = self.sam3.set_image_once(color_img)
        embed_elapsed = time.perf_counter() - embed_start

        self.timing_logger.log(
            frame=frame_count,
            class_name="ALL",
            stage="sam3_image_embedding",
            elapsed_seconds=embed_elapsed,
            num_masks=0,
            num_points=0
        )

        self.logger.info(
            f"TIMING frame={frame_count}, sam3_image_embedding_time={embed_elapsed:.4f}s"
        )

        for class_name in prompts:
            self.logger.info(
                f"Frame {frame_count}: running SAM3 for class '{class_name}'"
            )

            class_start_time = time.perf_counter()

            masks, boxes, scores = self.sam3.all_masks_from_state(
                image_state,
                class_name,
                score_threshold=confidence
            )

            class_elapsed = time.perf_counter() - class_start_time

            self.timing_logger.log(
                frame=frame_count,
                class_name=class_name,
                stage="sam3_class",
                elapsed_seconds=class_elapsed,
                num_masks=len(masks),
                num_points=0
            )

            self.logger.info(
                f"TIMING frame={frame_count}, class={class_name}, "
                f"sam3_time={class_elapsed:.2f}s, masks={len(masks)}"
            )

            if len(masks) == 0:
                self.logger.info(
                    f"Frame {frame_count}: no masks found for '{class_name}'"
                )
                continue

            class_mask = np.zeros_like(masks[0], dtype=np.uint8)

            for m in masks:
                class_mask = np.logical_or(class_mask, m).astype(np.uint8)

            if semantic_mask is None:
                semantic_mask = np.zeros_like(class_mask, dtype=np.uint8)

            semantic_mask = np.logical_or(semantic_mask, class_mask).astype(np.uint8)

            class_id = self.class_registry.get_class_id(class_name)
            color = self.class_registry.get_class_color(class_name)

            if class_mask.shape[:2] != semantic_map.shape[:2]:
                class_mask_resized = cv2.resize(
                    class_mask.astype(np.uint8),
                    (semantic_map.shape[1], semantic_map.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                class_mask_resized = class_mask

            semantic_map[class_mask_resized == 1] = class_id
            semantic_map_color[class_mask_resized == 1] = color

            semantic_overlay[class_mask == 1] = color

            for i, b in enumerate(boxes):
                if b is None:
                    continue

                s = None
                if scores is not None and len(scores) > i:
                    s = scores[i]

                all_boxes.append(b)
                all_scores.append(s)
                all_labels.append(class_name)

        frame_elapsed = time.perf_counter() - frame_start_time

        self.timing_logger.log(
            frame=frame_count,
            class_name="ALL",
            stage="sam3_full_frame",
            elapsed_seconds=frame_elapsed,
            num_masks=len(all_boxes),
            num_points=0
        )

        self.logger.info(
            f"TIMING frame={frame_count}, full_sam3_time={frame_elapsed:.2f}s"
        )

        self.logger.info(
            f"SAM3 done on frame {frame_count}: "
            f"mask_found={semantic_mask is not None}, "
            f"instances_found={len(all_boxes)}, "
            f"classes_found={list(set(all_labels))}"
        )

        return {
            "mask": semantic_mask,
            "box": all_boxes,
            "score": all_scores,
            "labels": all_labels,
            "semantic_overlay": semantic_overlay,
            "semantic_map": semantic_map,
            "semantic_map_color": semantic_map_color,
        }