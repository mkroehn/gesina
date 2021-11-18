import Camera
import Processors
import Display
import States
import Tick
import Config


def main():
    # flow control
    STATE = 0

    # instances
    conf = Config.Config()
    cam = Camera.Camera(conf)
    proc = Processors.Processors(conf)
    disp = Display.Display(conf)
    states = States.States()
    tick = Tick.Tick(4)

    # Start streaming
    cam.start()
    disp.start()

    try:
        while True:
            # state machine
            if STATE == 0:
                STATE = states.main_menu(disp, cam)
            if (STATE == 1) & cam.status:
                STATE = states.processing(disp, cam, proc, tick, conf)
            if (STATE == 2) & cam.status:
                STATE = states.calibration(disp, cam, proc, conf)
            if (STATE == 3) & cam.status:
                STATE = states.training(disp, proc, cam, conf)
            if (STATE == 4) & cam.status:
                STATE = states.livestream(disp, cam, proc)
            if STATE == -1:
                break

    finally:
        # Stop streaming
        disp.stop()
        cam.stop()


if __name__ == "__main__":
    # execute only if run as a script
    main()
