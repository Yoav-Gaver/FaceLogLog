import numpy as np
import socket

import rsa.key

from Faceloglog.face_hash import FaceHasher
import logging
import Protocol
import cv2
import msvcrt


class Client:
    def __init__(self, address=("127.0.0.1", 50_000), model="models/opencv/haarcascade_frontalface_alt2.xml",
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
        self.public_key: rsa.key.PublicKey | None = None

    def run(self):
        """run the client and send vectors to server"""
        self.connect_server()

        try:
            while msvcrt.kbhit() and msvcrt.getch() == b"q":
                _, frame = self.cam.read()

                vectors = self.vectorizer.get_faces_vectors(frame=frame)

                self.send_vectors(vectors)
        except KeyboardInterrupt as e:
            logging.info("client socket closed by keyboard interruption")
        Protocol.exit_socket(self.socket, self.public_key)
        self.socket.close()

    def connect_server(self):
        """connect socket to server and get public key"""
        logging.info("creating socket")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logging.info("contenting to server")
        self.socket.connect(self.address)
        logging.info("connected")
        self.public_key = Protocol.recv_key(self.socket)

    def send_vectors(self, vectors):
        """
        send all vectors to server

        Args:
            vectors: all the vectors to send
        """
        for v in vectors:
            Protocol.send(self.socket, v, self.public_key)


def main():
    client = Client()
    client.run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(levelname)%s %(asctime)s: %(message)s [%(module)s, %(funcName)s(%(lineno)d)]")
