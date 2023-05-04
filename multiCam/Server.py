import numpy as np
import socket
import select
import rsa
from Faceloglog import camera_counting
import logging
import Protocol
import threading
import msvcrt


class Server(threading.Thread):
    def __init__(self, lg_num_buckets=5, port=50_000):
        """
         Args:
            lg_num_buckets (int): the lg of the buckets that keep count(buckets=2^lg_num_buckets)
            port (int): the port the server will run in
        """
        self.counter = camera_counting.FaceCounter(lg_num_buckets=lg_num_buckets)
        self.port = port
        self.server_socket = None
        self.public_key, self.private_key = rsa.newkeys(1024)
        self.client_sockets = []
        self.running = threading.Event()

    def run(self):
        """
        runs the server and awaits clients to send vectors
        """
        self.server_socket = self.initiate_server()

        while self.running.is_set():
            rlist, _, xlist = select.select([self.server_socket] + self.client_sockets, [], [])

            data = self.receive_all(self.client_sockets, rlist)

            self.process_data(data, self.client_sockets)

        self.server_socket.close()

    def receive_all(self, client_sockets, rlist):
        data = []
        for current_socket in rlist:
            if current_socket is self.server_socket:
                self.add_client(client_sockets, current_socket)
            else:
                vector = Protocol.recv(current_socket, self.private_key)
                if vector is None:
                    try:
                        client_sockets.remove(current_socket)
                    except ValueError as e:
                        logging.debug(f"Connection already disconnected: {current_socket}\n{e}")
                else:
                    data.append(vector)

        return data

    def initiate_server(self):
        logging.info("Setting up server...")
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(("0.0.0.0", self.port))
        server_socket.listen()
        logging.info("Listening for clients...")

        return server_socket

    def add_client(self):
        connection, client_address = self.server_socket.accept()
        logging.info(f"New client joined! {client_address}")
        self.client_sockets.append(connection)
        Protocol.send_key(connection, self.public_key)


def main():
    server = Server()
    server.run()
    while msvcrt.getch() != b"q":
        pass
    server.join()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(levelname)%s %(asctime)s: %(message)s [%(module)s, %(funcName)s(%(lineno)d)]")
    main()
