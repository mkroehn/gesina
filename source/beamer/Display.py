import numpy as np
import cv2

class Display:
    # internal
    view_reduction = 0
    border_thickness = 5
    border_color = (200, 0, 0)
    padding = 5
    fullscreen = False

    def __init__(self, conf):
        self.vid_h = conf.vid_h
        self.vid_w = conf.vid_w
        self.img_h = conf.img_h
        self.img_w = conf.img_w
        self.insitu_img = np.zeros((conf.vid_h, conf.vid_w, 3), np.uint8)
        self.compression = conf.sampling_reduction
        self.view_reduction = conf.view_reduction
        self.fullscreen = conf.fullscreen

    def clear(self):
        cv2.rectangle(self.insitu_img,
                      pt1=(0, 0),
                      pt2=(self.vid_w, self.vid_h),
                      color=(0, 0, 0),
                      thickness=-1)

    def add_button(self, cx, cy, r, col):
        cv2.circle(self.insitu_img, center=(cx, cy), radius=r, color=col, thickness=-1)

    def add_border(self, cx, cy, w, h):
        cv2.rectangle(self.insitu_img,
                      pt1=(cx, cy),
                      pt2=(int((cx+w)/self.view_reduction) + 2*self.padding, int((cy+h)/self.view_reduction) + 2*self.padding),
                      color=self.border_color,
                      thickness=self.border_thickness)

    def update_streams(self, depth_img, color_img):
        reduced_color = color_img[0:color_img.shape[0]:self.view_reduction, 0:color_img.shape[1]:self.view_reduction, :]
        reduced_depth = depth_img[0:depth_img.shape[0]:self.view_reduction, 0:depth_img.shape[1]:self.view_reduction, :]
        images = np.hstack((reduced_color, reduced_depth))
        self.insitu_img[self.padding:images.shape[0]+self.padding, self.padding:images.shape[1]+self.padding, :] = images

    def update_info(self, info):
        img_info = np.zeros((20, 400, 3))
        cv2.putText(img_info, text=info, org=(0, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(200, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        self.insitu_img[140:160, 10:410, :] = img_info

    def add_static_text(self, txt, xpos, ypos, color, scale):
        cv2.putText(self.insitu_img, text=txt, org=(xpos, ypos), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale, color=color, thickness=1, lineType=cv2.LINE_AA)

    def show(self):
        if self.fullscreen:
            cv2.namedWindow('RealSense', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('RealSense', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', self.insitu_img)
        return cv2.waitKey(1)

    def color_depth_from_frame(self, depth_frame):
        depth_image = np.asanyarray(depth_frame.get_data())[0:self.img_h:self.compression, 0:self.img_w:self.compression]
        return cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_PINK)

    def color_depth_from_image(self, depth_image):
        return cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_PINK)

    def start(self):
        self.show()

    def stop(self):
        cv2.destroyAllWindows()