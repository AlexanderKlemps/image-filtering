import time
import os
import cv2
import sys
import numpy as np
from filters.image_filters import PeronaMalikFilter, OsmosisFilter, CoherenceShockFilter
from filters.utils import resize, put_text

FILTER_DICT = {
    "PeronaMalikFilter": PeronaMalikFilter,
    "OsmosisFilter": OsmosisFilter,
    "CoherenceFilter": CoherenceShockFilter
}


class FilterEditor(object):

    def __init__(self, filter_name):
        self.filter_name = filter_name

        cv2.namedWindow(self.filter_name, 1)
        cv2.moveWindow(self.filter_name, 0, 0)

        self.filter = FILTER_DICT[self.filter_name]()
        for key, value in self.filter.parameters.items():
            max_value = self.filter.max_values[key]
            cv2.createTrackbar(key, self.filter_name, int(value * 10), int(max_value * 10), lambda x: x)

        self.source_image = cv2.imread("Resources/lena.jpg")
        self.target_shape = max(self.source_image.shape)
        self.process_image = self.source_image.copy()
        self.control_width = 400
        self.control_image = []

    def display_controls(self):
        control_info_strings = [
            "Controls:",
            " 'Q' - Quit",
            " 'P' - Process",
            " 'S' - Save",
            " 'C' - Change Filter",
            "",
            "Current config:"
        ]

        for key, value in self.filter.parameters.items():
            control_info_strings.append(" {} = {:.2f}".format(key, value))

        control_info_shape = list(self.process_image.shape)
        control_info_shape[1] = self.control_width
        control_info = np.zeros(control_info_shape, dtype=self.process_image.dtype)
        for idx, control_str in enumerate(control_info_strings):
            put_text(control_info, text=control_str, pos=(0, (idx + 1) * 30))
        self.process_image = np.hstack((control_info, self.process_image))

    def run(self, image_input=None):
        if image_input is not None:
            max_dim = max(image_input.shape)
            f = self.target_shape / max_dim
            self.source_image = resize(image_input, scale=f)
        self.process_image = self.source_image.copy()
        self.display_controls()

        while True:
            cv2.imshow(self.filter_name, self.process_image)

            key = cv2.waitKey(1)
            if key == ord('q'):
                sys.exit(0)
            if key == ord("p"):
                self.update()
            if key == ord("s"):
                self.save()
            if key == ord("c"):
                cv2.destroyWindow(self.filter_name)
                break

    def update(self):
        for key in self.filter.parameters.keys():
            pos = max(1, cv2.getTrackbarPos(key, self.filter_name)) / 10.0
            self.filter.parameters[key] = pos
        self.process_image = self.filter.process(self.source_image)
        self.display_controls()

    def save(self):
        base_dir = "Resources/{}".format(self.filter_name)
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        file_name = "{}/{}.png".format(base_dir, int(time.time()))
        print("Saving image to ./{}.".format(file_name))
        cv2.imwrite(file_name, self.process_image[:, self.control_width:, :])


if __name__ == '__main__':
    filter_keys = list(FILTER_DICT.keys())
    current_idx = 0
    while True:
        key = filter_keys[current_idx]
        editor = FilterEditor(key)
        editor.run()
        current_idx = (current_idx + 1) % len(filter_keys)
