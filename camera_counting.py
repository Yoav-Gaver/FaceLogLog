import time
import sys
import os
from tqdm import tqdm
import cv2
import urllib.request as urlreq
import numpy as np
from Faceloglog.face_hash import FaceHasher
from Faceloglog.hash_functions import LogLog
import logging
import msvcrt


# save face detection algorithm's url in haarcascade_url variable
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

# save face detection algorithm's name as haarcascade
haarcascade = "models/opencv/haarcascade_frontalface_alt2.xml"

# # save facial landmark detection model's url in LBFmodel_url variable
# LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
#
# # save facial landmark detection model's name as LBFmodel
# LBFmodel = "models/opencv/lbfmodel.yaml"

# Flags for the program
FLAGS = ["-q", "--quiet", "-c", "--camera", "-h", "--help", "-d", "--debug", "-i", "--init"]


def initiate():
    # check if file is in needed directory
    if haarcascade in os.listdir(os.curdir):
        print("Haarcascade model exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
        urlreq.urlretrieve(haarcascade_url, haarcascade)
        print("Haarcascade model downloaded")
    #
    # # check if file is in needed directory
    # if LBFmodel in os.listdir(os.curdir):
    #     print("LBF model exists")
    # else:
    #     # download picture from url and save locally as lbfmodel.yaml, < 54MB
    #     urlreq.urlretrieve(LBFmodel_url, LBFmodel)
    #     print("LBF model downloaded")


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
        if not use_camera:
            dir_path = input("what path is the images in?")
    except KeyboardInterrupt:
        logging.critical("program ended through user interruption")
        return
    if not lg_buckets.isnumeric():
        logging.critical("inputted string must be an integer")
        return
    counter = FaceCounter(lg_num_buckets=int(lg_buckets), show_images=show_images)

    if use_camera:
        cam = cv2.VideoCapture(0)
        while True:
            _, frame = cam.read()

            process_frame(counter, frame, show_images)

            if frame is None:
                logging.critical("image empty check source")
                break
            
            key = cv2.waitKey(1) & 0xFF
            if key != 0:
                if key == ord('q'):
                    break
                if key == ord('e'):
                    print(f"buckets:{counter.loglog}\nestimated number of unique faces seen: {counter.estimate()}")

            if msvcrt.kbhit():
                if msvcrt.getch() == b'q':
                    break
                if msvcrt.getch() == b'e':
                    print(f"buckets:{counter.loglog}\nestimated number of unique faces seen: {counter.estimate()}")

        # After the loop release the cap object
        cam.release()
    elif dir_path:
        for path in tqdm(os.listdir(dir_path)):
            frame = cv2.imread(f"{dir_path}\\{path}")
            process_frame(counter, frame, show_images)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    else:
        logging.critical("Something went wrong")

    print(f"buckets:{counter.loglog}\nestimated face seen: {counter.estimate()}")


def process_frame(counter, frame, show_images):
    ptime = time.time()

    counter.add_faces(frame=frame)

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
        elif flag in ["-c", "--camera"]:  # use the camera instid of folder
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
