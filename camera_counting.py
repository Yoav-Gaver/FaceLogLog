import time
import sys
import os
from tqdm import tqdm
import cv2
import numpy as np
from Faceloglog import FaceHasher, LogLog, initiate
import logging
import msvcrt


# Flags for the program
FLAGS = ["-q", "--quiet", "-c", "--camera", "-h", "--help", "-d", "--debug", "-i", "--init"]



class FaceCounter:
    def __init__(self, lg_num_buckets=5, model="models/opencv/haarcascade_frontalface_alt2.xml",
                 face_size=(256, 256), show_images=False):
        """
        creates a face counter object

        Args:
            lg_num_buckets (int): the lg of the buckets that keep count(buckets = 2^lg_num_buckets)
            model (str): the path of the desired face detection algorithm
            face_size (tuple[int]): the size that the face will be stretched and cut to
            show_images (bool): helps to identify problems in the model
        """
        self.lg_num_bucket = lg_num_buckets
        self.model = model
        self.face_size = face_size

        self.loglog = LogLog(lg_num_buckets=self.lg_num_bucket)
        self.face_vectorizer = FaceHasher(model=model, face_size=face_size, show_images=show_images)

    def get_bucket_ind(self, vector):
        """
        return the bucket the vector belongs to

        Args:
             vector (np.ndarray): the vector

         Returns:
             the index of the bucket
        """
        bin_bucket = vector[-self.lg_num_bucket:]

        bucket_ind = 0
        for bit in bin_bucket:
            bucket_ind = bucket_ind * 2 + bit

        return bucket_ind

    def add_faces(self, frame):
        """
        adds all the faces to the buckets they are supposed to be in

        Args:
            frame (np.ndarray): the frame with the faces
        """
        vectors = self.face_vectorizer.get_faces_vectors(frame=frame)
        for v in vectors:
            logging.debug(v)

        for ind, v in enumerate(vectors):
            self.add_vector(v, ind)
        return vectors

    def add_vector(self, v, ind=None):
        zeros = FaceHasher.leading_zeros(v, self.lg_num_bucket)
        bucket_ind = self.get_bucket_ind(v)
        if ind:
            logging.debug(f"face number {ind}: {v}")
        logging.debug(f"zeros: {zeros}, to bucket: {bucket_ind}")
        self.loglog.add_by_zeros(zeros=zeros, bucket_ind=bucket_ind)

    def estimate(self, correction_m: int = 1, use_hyper: bool = False):
        if use_hyper:
            return self.loglog.hyper_estimate(correction_m=correction_m)
        return self.loglog.estimate(correction_m=correction_m)


def main(args: list):
    show_images, use_camera = args

    try:
        lg_buckets = input("What would you like the lg of number of buckets to be?(2=>4, 3=>8,...,10=>1024,...)\n")
        lg_buckets = int(lg_buckets)

        if not use_camera:
            dir_path = input("what path is the images in?")
    except KeyboardInterrupt:
        logging.critical("program ended through user interruption")
        return
    except ValueError:
        logging.critical("inputted string must be an integer")
        return

    main_counter = FaceCounter(lg_num_buckets=lg_buckets, show_images=show_images)

    lglg = int(np.ceil(np.log2(lg_buckets)))
    counter_array = []
    for i in range(lg_buckets - lglg, lg_buckets + lglg + 1):
        if i != lg_buckets:
            counter_array.append(FaceCounter(lg_num_buckets=i, show_images=show_images))
            logging.info(f"added a counter with 2^{i} buckets")
    logging.debug(f"counter array length{len(counter_array)}")

    if use_camera:
        cam = cv2.VideoCapture(0)
        while True:
            _, frame = cam.read()

            process_frame(main_counter=main_counter, frame=frame,
                          show_images=show_images, counter_array=counter_array)

            if frame is None:
                logging.critical("image empty check source")
                break

            key = cv2.waitKey(1) & 0xFF
            if key != 0:
                if key == ord('q'):
                    break
                if key == ord('e'):
                    print_counter(main_counter=main_counter, counter_array=counter_array)

            if msvcrt.kbhit():
                if msvcrt.getch() == b'q':
                    break
                if msvcrt.getch() == b'e':
                    print_counter(main_counter=main_counter, counter_array=counter_array)
        # After the loop release the cap object
        cam.release()
    elif dir_path:
        for path in tqdm(os.listdir(dir_path)):
            frame = cv2.imread(f"{dir_path}\\{path}")
            process_frame(main_counter, frame, show_images, counter_array=counter_array)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if msvcrt.kbhit():
                if msvcrt.getch() == b'q':
                    break
    else:
        logging.critical("Something went wrong")

    print_counter(main_counter=main_counter, counter_array=counter_array)


def print_counter(main_counter, counter_array=None):
    """
    print the counter information for all buckets and different counters

    Args:
        main_counter (FaceCounter): the main counter
        counter_array (list): a list of counters with different buckets
    """
    print(f"\n\nmain counter:\nbuckets:{main_counter.loglog}\nestimated face seen: {main_counter.estimate()}")

    bucket_numbers = []
    estimates = []
    for counter in counter_array:
        bucket_numbers.append(counter.lg_num_bucket)
        estimates.append(counter.estimate())

    print(f"\nnear counter:\n"
          f"estimates of different counters of different bucket numbers {estimates}"
          f"\nlg of the bucket numbers of the counters are {bucket_numbers}"
          f"\naverage of all estimates {np.average(estimates)}")


def process_frame(main_counter, frame, show_images, counter_array=None):
    """
    Process a frame and add it to the counters.

    Args:
        main_counter (FaceCounter): the main counter
        frame (np.ndarray): the frame to process
        show_images (bool): whether to show images
        counter_array (list): a list of counters with different buckets

    """
    ptime = time.time()

    faces = main_counter.add_faces(frame=frame)

    if counter_array:
        for face in faces:
            for counter in counter_array:
                counter.add_vector(face)

    if show_images:
        # add fps to image
        ctime = time.time()
        fps = 1 / (ctime - ptime)

        cv2.putText(frame, f"FPS: {fps:,.1f}", (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)

        cv2.imshow("frame", frame)


def usage():
    print("This program is used for counting unique faces in frames.\n"
          "It uses the Hyperloglog algorithm and a face vectorization algorithm to"
          "count them. press 'e' to get estimated count and 'q' to exit.\n\n"
          "You can use this flags:"
          "use '-q' or '--quiet' to hide the frames. \n"
          "use '-c' or '--camera' to use camera instead.\n"
          "use '-d' or '--debug to show more information of the program IRT.\n"
          "use '-h' or '--help' to get all commands again.")


def handle_flags():
    args = sys.argv[1:]
    show_images, use_camera, output_help, debug, init = (True, False, False, False, False)
    for flag in args:
        if flag not in FLAGS:
            logging.warn(f"Unrecognized flag {flag}\n\n")
            continue

        if flag in ["-q", "--quiet"]:  # quiet mode
            show_images = False
        elif flag in ["-c", "--camera"]:  # use the camera instead of folder
            use_camera = True
        elif flag in ["-h", "--help"]:  # print help
            output_help = True
        elif flag in ["-d", "--debug"]:  # debug mode
            debug = True
        elif flag in ["-i", "--init"]:  # initialize needed files
            init = True

    return show_images, use_camera, debug, init, output_help


if __name__ == '__main__':
    # run the program
    flags = handle_flags()
    if not flags[-1]:
        logging.basicConfig(level=logging.DEBUG if flags[2] else logging.INFO,
                            format="%(levelname)s %(asctime)s: %(message)s [%(module)s, %(funcName)s(%(lineno)d)]")
        if flags[3]:
            initiate()

        main(flags[:2])
    else:
        usage()
    # Destroy all the windows
    cv2.destroyAllWindows()
