import socket
import select
import logging
import Protocol
import threading
import msvcrt
from camera_counting import FaceCounter


class Server:
    def __init__(self, lg_num_buckets=5, port=Protocol.PORT):
        """
         Args:
            lg_num_buckets (int): the lg of the buckets that keep count(buckets=2^lg_num_buckets)
            port (int): the port the server will connect to
        """
        self.counter = FaceCounter(lg_num_buckets=lg_num_buckets)
        self.address = ("127.0.0.1", port)
        self.server_socket = None
        self.client_sockets = []
        self.running = threading.Event()

    def run(self):
        """
        runs the server and awaits clients to send vectors
        """
        self.initiate_server()

        print("Press 'e' to get estimate of how many people were seen by all cameras.\n"
              "Hold 'q' to end server program.")

        while True:
            rlist, _, xlist = select.select([self.server_socket] + self.client_sockets, self.client_sockets, [], 1)

            data = self.receive_all(rlist)

            if len(data):
                self.process_data(data)

            if msvcrt.kbhit():
                if msvcrt.getch() == b'q':
                    break
                if msvcrt.getch() == b'e':
                    print(f"estimated face seen: {self.estimate()}")

        print(f"current estimate of unique faces seen is {self.estimate()}")
        logging.debug("closing")
        self.server_socket.close()

    def receive_all(self, rlist):
        data = []
        for current_socket in rlist:
            if current_socket is self.server_socket:
                self.add_client()
            else:
                vector = Protocol.recv(current_socket)
                logging.debug(vector)
                if vector is None:
                    try:
                        self.client_sockets.remove(current_socket)
                        logging.debug("client socket removed")
                    except ValueError:
                        logging.debug(f"Connection already disconnected: {current_socket}\n")
                else:
                    data.append(vector)

        return data

    def estimate(self):
        return self.counter.estimate()

    def initiate_server(self):
        logging.info("Setting up server...")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(self.address)
        self.server_socket.listen()
        logging.info("Listening for clients...")

    def add_client(self):
        connection, client_address = self.server_socket.accept()
        logging.info(f"New client joined! {client_address}")
        self.client_sockets.append(connection)

    def process_data(self, data):
        for vector in data:
            self.counter.add_vector(vector)


def main():
    try:
        lg_buckets = input("What would you like the lg of number of buckets to be?(2=>4, 3=>8,...,10=>1024,...)\n")
    except KeyboardInterrupt:
        logging.critical("program ended through user interruption")
        return

    if not lg_buckets.isnumeric():
        logging.critical("inputted string must be an integer")
        return
    server = Server(lg_num_buckets=int(lg_buckets))
    server.run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(asctime)s: %(message)s [%(module)s, %(funcName)s(%(lineno)d)]",
                        datefmt="%H:%M:%S")
    main()
