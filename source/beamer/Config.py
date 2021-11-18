class Config:
    # debug output
    DEBUG = False
    DEBUG_PATH = './dump/'
    TRAIN_PATH = './data/right_singletap/'

    # Processing Variables
    vid_h, vid_w = (430, 790)
    sampling_reduction = 2
    view_reduction = 1
    sleeptime = 0.5

    # Processor: BG Separation
    clipping_tolerance = 10

    # Processor: Min Distance
    min_distance = 200

    # Processor: Brightness
    brightness_limit = 0

    # Camera Parameters
    img_h, img_w = (480, 640)
    fps = 30
    train_loops = 100
    train_region = 10
    train_frames = 15
    calib_brightness_limit = 0.7
    calib_separation_limit = 0.15
    calib_file_x = './calib_x.csv'
    calib_file_y = './calib_y.csv'

    # Display
    fullscreen = False
