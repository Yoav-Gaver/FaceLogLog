import sys
import socket
from Faceloglog import FaceHasher
import logging
import Protocol
import cv2
import msvcrt


class Client:
    def __init__(self, address=("127.0.0.1", Protocol.PORT), model="models/opencv/haarcascade_frontalface_alt2.xml",
                 face_size=(256, 256), show_images=False, video_capture=0):
        """
        creates a client that vecterizes faces from the camera and sends the vectors to a server

        Args:
            address ((str, int)): the address of the server
            model (str): the path of the desired face detection algorithm
            face_size (tuple[int]): the size that the face will be stretched and cut to
            show_images (bool): helps to identify problems in the model
            video_capture (str|int): the camera that the client uses
        """
        self.address = address
        self.vectorizer = FaceHasher(model=model, face_size=face_size,
                                     show_images=show_images)
        self.cam = cv2.VideoCapture(video_capture)
        self.socket: socket.socket | None = None

    def run(self):
        """run the client and send vectors to server"""
        self.connect_server()

        while not msvcrt.kbhit() or msvcrt.getch() == b"q":
            _, frame = self.cam.read()
            cv2.imshow("frame", frame)

            vectors = self.vectorizer.get_faces_vectors(frame=frame)

            if len(vectors):
                try:
                    logging.debug(vectors)
                    self.send_vectors(vectors)
                except ConnectionAbortedError:
                    logging.warning("Connection was forcefully closed by server. Exiting program...")
                    self.socket.close()
                    return

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        Protocol.exit_socket(self.socket)
        self.socket.close()

    def connect_server(self):
        """connect socket to server and get public key"""
        logging.debug("Creating socket")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logging.debug("Contenting to server")
        self.socket.connect(self.address)
        logging.debug("Connected")

    def send_vectors(self, vectors):
        """
        send all vectors to server

        Args:
            vectors: all the vectors to send
        """
        for v in vectors:
            Protocol.send(self.socket, v)


def main():
    if len(sys.argv) == 2:
        client = Client(address=(sys.argv[1], Protocol.PORT))
    else:
        client = Client()
    client.run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(asctime)s: %(message)s [%(module)s, %(funcName)s(%(lineno)d)]")
    main()
