import csv
import os
import time
from datetime import datetime

class TimingLogger:
    def __init__(self, output_dir="timing_results"):
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(output_dir, f"sam3_timing_{timestamp}.csv")

        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "frame",
            "class_name",
            "stage",
            "elapsed_seconds",
            "num_masks",
            "num_points",
            "fps",
            "vram_mb",
            "num_classes"
        ])
        self.file.flush()

    def log(
        self,
        frame,
        class_name,
        stage,
        elapsed_seconds,
        num_masks=0,
        num_points=0,
        fps=0.0,
        vram_mb=0.0,
        num_classes=0
    ):
        self.writer.writerow([
            frame,
            class_name,
            stage,
            round(elapsed_seconds, 4),
            num_masks,
            num_points,
            round(fps, 2),
            round(vram_mb, 1),
            num_classes
        ])
        self.file.flush()

    def close(self):
        self.file.close()