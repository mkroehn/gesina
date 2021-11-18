import numpy as np
import cv2


class Processors:
    h, w = (480, 640)
    kernel_size = 5
    iter = 4

    def __init__(self, conf):
        self.compression = conf.sampling_reduction
        self.clipping_tolerance = conf.clipping_tolerance
        self.brightness_level = conf.brightness_limit
        self.min_distance = conf.min_distance

    def generate_background(self, depth_frame):
        result_img = \
            np.asanyarray(depth_frame.get_data())[0:self.h:self.compression, 0:self.w:self.compression] - \
            self.clipping_tolerance
        print(result_img.shape)
        result_img = cv2.erode(result_img, np.ones((self.kernel_size, self.kernel_size), np.uint8),
                               iterations=self.iter)
        return np.dstack((result_img, result_img, result_img))

    def process(self, depth_frame, color_frame, bg_image_3d):
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())[0:self.h:self.compression, 0:self.w:self.compression]
        color_image = np.asanyarray(color_frame.get_data())[0:self.h:self.compression, 0:self.w:self.compression, :]

        # Background Separation
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        result_img = np.where(depth_image_3d < bg_image_3d, color_image, 0)
        result_depth_3d = np.where(depth_image_3d < bg_image_3d, depth_image_3d, 0)

        # Minimum Distance
        result_img = np.where(depth_image_3d > self.min_distance, result_img, 0)

        # Only the brightest channels survive
        result_img = np.where(result_img > self.brightness_level, result_img, 0)

        return result_img, result_depth_3d

    def blob_detection(self, cam, conf, background_image):
        color_image = np.sum(cam.get_color_image(), axis=2) / 3 / 255
        diff_img = np.where(np.abs(color_image - background_image) > conf.calib_separation_limit, color_image, 0)
        index = np.where(diff_img > conf.calib_brightness_limit)
        y = int(np.sum(index[0] / index[0].shape[0]))
        x = int(np.sum(index[1] / index[1].shape[0]))
        return x, y, color_image, diff_img
