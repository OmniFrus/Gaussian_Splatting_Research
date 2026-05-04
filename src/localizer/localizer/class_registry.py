import cv2
import numpy as np

class ClassRegistry:
    def __init__(self):
        self.class_to_id = {"background": 0}
        self.next_class_id = 1

        self.default_classes = [
            "chair",
            "table",
            "person",
            "monitor",
            "keyboard",
            "mouse",
            "floor",
            "wall"
        ]

    def get_class_id(self, class_name):
        if class_name not in self.class_to_id:
            self.class_to_id[class_name] = self.next_class_id
            self.next_class_id += 1

        return self.class_to_id[class_name]

    def get_class_color(self, class_name):
        class_id = self.get_class_id(class_name)
        hue = (class_id * 40) % 180

        color = cv2.cvtColor(
            np.uint8([[[hue, 255, 255]]]),
            cv2.COLOR_HSV2BGR
        )[0][0]

        return tuple(int(c) for c in color)