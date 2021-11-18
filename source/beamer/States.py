import numpy as np
import time
import random as rnd


class States:

    # State 0 - Main Menu
    def main_menu(self, disp, cam):
        # static video content
        disp.clear()
        disp.add_static_text('1: Run', 50, 100, (0, 200, 0), 1)
        disp.add_static_text('2: Calibration', 50, 150, (0, 200, 0), 1)
        if cam.calib_done:
            disp.add_static_text('(data available)', 300, 150, (200, 0, 0), 1)
        else:
            disp.add_static_text('(no calib data)', 300, 150, (0, 0, 200), 1)
        disp.add_static_text('3: Training', 50, 200, (0, 200, 0), 1)
        disp.add_static_text('4: Livestream', 50, 250, (0, 200, 0), 1)
        disp.add_static_text('0: Exit', 50, 350, (0, 200, 0), 1)

        while True:
            key = disp.show()
            if key == 27:
                return -1
            if key == ord('0'):
                return -1
            if key == ord('1'):
                return 1
            if key == ord('2'):
                return 2
            if key == ord('3'):
                return 3
            if key == ord('4'):
                return 4
            if key == -1:
                continue

    # State 1 - Processing [aka Run]
    def processing(self, disp, cam, proc, tick, conf):
        # static video content
        disp.clear()
        disp.add_border(0, 0, int(2 * conf.vid_w / conf.sampling_reduction), int(conf.vid_h / conf.sampling_reduction))
        disp.add_button(420, 70, 60, (0, 0, 200))
        disp.add_button(570, 70, 60, (0, 200, 0))
        disp.add_button(720, 70, 60, (200, 0, 0))

        # flag for background
        do_bg = True

        try:
            while True:
                # timer start for fps
                tick.tick(0)

                # get frames
                depth_frame, color_frame = cam.get_frames()

                # get timer for fps
                tick.tick(1)

                # Generate Background if not yet set
                if do_bg:
                    background_img = proc.generate_background(depth_frame)
                    do_bg = False

                # do processing
                result_img, result_depth_3d = proc.process(depth_frame, color_frame, background_img)
                depth_colormap = disp.color_depth_from_frame(depth_frame)

                # proc timer for fps
                tick.tick(2)

                # prepare videoframe
                disp.update_streams(depth_colormap, result_img)
                disp.update_info(tick.get_info())

                if disp.show() == 27:
                    break

                # update timer and fps
                tick.tick(3)
                tick.update_info()

        except:
            print('An error occurred in processing')
            return -1

        finally:
            return 0

    # State 2 - Calibration [fullres, no compression considered]
    def calibration(self, disp, cam, proc, conf):
        # reset lookup table
        cam.reset_lookup_table()
        # background
        disp.clear()
        disp.show()
        time.sleep(conf.sleeptime)
        background_image = np.sum(cam.get_color_image(), axis=2) / 3 / 255

        # display and evaluate dot images
        for i in range(0, 3):
            for j in range(0, 3):
                disp.clear()
                vid_x = i * 350 + 25
                vid_y = j * 175 + 25
                disp.add_button(vid_x, vid_y, 20, (255, 255, 255))
                disp.show()
                time.sleep(conf.sleeptime)
                cam_x, cam_y, img, diff = proc.blob_detection(cam, conf, background_image)

                if conf.DEBUG:
                    outfile = str(vid_x) + '_' + str(vid_y) + '.csv'
                    np.savetxt(conf.DEBUG_PATH + outfile, img)
                    np.savetxt(conf.DEBUG_PATH + 'diff-' + outfile, diff)
                    np.savetxt(conf.DEBUG_PATH + 'background.csv', background_image)

                cam.set_lookup_point(vid_x, vid_y, cam_x, cam_y)

        cam.do_lookup_table()
        cam.write_lookup_table()

        #y_test, x_test = cam.query_lookup_table(375, 200)
        #print(x_test, y_test)

        return 0

    # State 3 - Training
    def training(self, disp, proc, cam, conf):
        # background
        disp.clear()
        disp.show()
        depth_frame, color_frame = cam.get_frames()
        background_img = proc.generate_background(depth_frame)

        counter = 0
        while counter < conf.train_loops:
            #blank
            disp.clear()
            disp.add_static_text(str(counter), 50, 100, (0, 200, 0), 1)
            disp.show()
            time.sleep(conf.sleeptime)
            #blob
            vid_x = rnd.randrange(100, conf.vid_w - 100)
            vid_y = rnd.randrange(100, conf.vid_h - 100)
            cam_pts = cam.query_lookup_table(vid_x, vid_y)
            # consider compression
            cam_x = int(cam_pts[1] / conf.sampling_reduction)
            cam_y = int(cam_pts[0] / conf.sampling_reduction)
            disp.add_button(vid_x, vid_y, 60, (0, 0, 200))
            disp.show()
            time.sleep(conf.sleeptime)
            #record
            framecounter = 0
            result = np.zeros(shape=(2 * conf.train_region * conf.train_frames, 2 * conf.train_region))
            while framecounter < conf.train_frames:
                depth_frame, color_frame = cam.get_frames()
                # get result images, where compression is applied and background is separated
                result_img, result_depth_3d = proc.process(depth_frame, color_frame, background_img)
                result[2 * conf.train_region * framecounter:2 * conf.train_region * (framecounter+1), :] = result_depth_3d[cam_y-conf.train_region:cam_y+conf.train_region, cam_x-conf.train_region:cam_x+conf.train_region, 0]
                framecounter = framecounter + 1
            # write to file
            outfile = conf.TRAIN_PATH + str(counter) + '.csv'
            np.savetxt(outfile, result, delimiter=';')
            counter += 1
        return 0

    # State 4 - Live Stream
    def livestream(self, disp, cam, proc):
        try:
            disp.clear()
            while True:
                # get images
                depth_image, color_image = cam.get_images()
                depth_colormap = disp.color_depth_from_image(depth_image)

                # prepare videoframe
                disp.update_streams(depth_colormap, color_image)

                if disp.show() == 27:
                    break

        finally:
            return 0
