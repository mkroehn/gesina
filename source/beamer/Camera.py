import pyrealsense2 as rs
import numpy as np


def file_exists(filename):
    try:
        f = open(filename)
        f.close()
        return True
    except IOError:
        return False


class Camera:

    def __init__(self, conf):
        self.vid_h = conf.vid_h
        self.vid_w = conf.vid_w
        self.calib_file_x = conf.calib_file_x
        self.calib_file_y = conf.calib_file_y
        self.calib_done = False

        self.reset_lookup_table()
        self.read_lookup_table()

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.status = False

    def start(self):
        try:
            self.pipeline.start(self.config)
            self.status = True
        except:
            self.status = False
            print('Error: No Realsense Camera detected!')

    def stop(self):
        if self.status:
            self.pipeline.stop()
            self.status = False

    def get_frames(self):
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            break

        return depth_frame, color_frame

    def get_depth_frame(self):
        depth_frame, color_frame = self.get_frames()
        return depth_frame

    def get_color_frame(self):
        depth_frame, color_frame = self.get_frames()
        return color_frame

    def get_images(self):
        depth_frame, color_frame = self.get_frames()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image

    def get_depth_image(self):
        depth_image, color_image = self.get_images()
        return depth_image

    def get_color_image(self):
        depth_image, color_image = self.get_images()
        return color_image

    def set_lookup_point(self, x, y, x_img, y_img):
        self.lookup_table[y, x, :] = (y_img, x_img)

    def do_lookup_table(self):
        index = np.where(self.lookup_table > 0)
        x_key = (index[1].min(), int(index[1].sum() / 6 - index[1].min() - index[1].max()), index[1].max())
        y_key = (index[0].min(), int(index[0].sum() / 6 - index[0].min() - index[0].max()), index[0].max())

        for i in range(0, len(x_key)):
            for j in range(0, len(y_key)):
                if i < 2:
                    dx = (self.lookup_table[y_key[j], x_key[i + 1], 1] - self.lookup_table[y_key[j], x_key[i], 1]) / (
                            x_key[i + 1] - x_key[i])
                    dy = (self.lookup_table[y_key[j], x_key[i + 1], 0] - self.lookup_table[y_key[j], x_key[i], 0]) / (
                            x_key[i + 1] - x_key[i])
                    for h in range(1, x_key[i + 1] - x_key[i]):
                        self.lookup_table[y_key[j], x_key[i] + h, 1] = self.lookup_table[y_key[j], x_key[i], 1] + dx * h
                        self.lookup_table[y_key[j], x_key[i] + h, 0] = self.lookup_table[y_key[j], x_key[i], 0] + dy * h
                if j < 2:
                    dx = (self.lookup_table[y_key[j + 1], x_key[i], 1] - self.lookup_table[y_key[j], x_key[i], 1]) / (
                            y_key[j + 1] - y_key[j])
                    dy = (self.lookup_table[y_key[j + 1], x_key[i], 0] - self.lookup_table[y_key[j], x_key[i], 0]) / (
                            y_key[j + 1] - y_key[j])
                    for h in range(1, y_key[j + 1] - y_key[j]):
                        self.lookup_table[y_key[j] + h, x_key[i], 1] = self.lookup_table[y_key[j], x_key[i], 1] + dx * h
                        self.lookup_table[y_key[j] + h, x_key[i], 0] = self.lookup_table[y_key[j], x_key[i], 0] + dy * h

        for i in range(x_key[0] + 1, x_key[2]):
            if i == x_key[1]:
                continue
            for j in range(0, 2):
                dx = (self.lookup_table[y_key[j + 1], i, 1] - self.lookup_table[y_key[j], i, 1]) / (
                            y_key[j + 1] - y_key[j])
                dy = (self.lookup_table[y_key[j + 1], i, 0] - self.lookup_table[y_key[j], i, 0]) / (
                            y_key[j + 1] - y_key[j])
                for h in range(1, y_key[j + 1] - y_key[j]):
                    self.lookup_table[y_key[j] + h, i, 1] = self.lookup_table[y_key[j], i, 1] + dx * h
                    self.lookup_table[y_key[j] + h, i, 0] = self.lookup_table[y_key[j], i, 0] + dy * h

        for j in range(y_key[0] + 1, y_key[2]):
            if j == y_key[1]:
                continue
            for i in range(0, 2):
                dx = (self.lookup_table[j, x_key[i + 1], 1] - self.lookup_table[j, x_key[i], 1]) / (
                            x_key[i + 1] - x_key[i])
                dy = (self.lookup_table[j, x_key[i + 1], 0] - self.lookup_table[j, x_key[i], 0]) / (
                            x_key[i + 1] - x_key[i])
                for h in range(1, x_key[i + 1] - x_key[i]):
                    self.lookup_table[j, x_key[i] + h, 1] = self.lookup_table[j, x_key[i], 1] + dx * h
                    self.lookup_table[j, x_key[i] + h, 0] = self.lookup_table[j, x_key[i], 0] + dy * h

        self.calib_done = True

    def query_lookup_table(self, x, y):
        return self.lookup_table[y, x, :]

    def write_lookup_table(self):
        if self.calib_done:
            np.savetxt(self.calib_file_x, self.lookup_table[:, :, 1], delimiter=';')
            np.savetxt(self.calib_file_y, self.lookup_table[:, :, 0], delimiter=';')

    def read_lookup_table(self):
        if file_exists(self.calib_file_x):
            self.lookup_table[:, :, 1] = np.genfromtxt(self.calib_file_x, delimiter=';')
            if file_exists(self.calib_file_y):
                self.lookup_table[:, :, 0] = np.genfromtxt(self.calib_file_y, delimiter=';')
                self.calib_done = True

    def reset_lookup_table(self):
        self.lookup_table = np.zeros(shape=(self.vid_h, self.vid_w, 2))
