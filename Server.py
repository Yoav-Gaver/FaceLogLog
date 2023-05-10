import socket
import select
import logging
import Protocol
import threading
import msvcrt
import camera_counting


class Server:
    def __init__(self, lg_num_buckets=5, port=Protocol.PORT):
        """
         Args:
            lg_num_buckets (int): the lg of the buckets that keep count(buckets=2^lg_num_buckets)
            port (int): the port the server will connect to
        """
        self.counter = camera_counting.FaceCounter(lg_num_buckets=lg_num_buckets)
        self.address = ("127.0.0.1", port)
        self.server_socket = None
        self.client_sockets = []
        self.running = threading.Event()

    def run(self):
        """
        runs the server and awaits clients to send vectors
        """
        self.initiate_server()

        logging.info("Press 'e' to get estimate of how many people were seen by all cameras.\n"
                     "Hold 'q' to end server program.")

        while not msvcrt.kbhit() or msvcrt.getch() != b"q":
            logging.debug("getting data")
            rlist, _, xlist = select.select([self.server_socket] + self.client_sockets, self.client_sockets, [], 1)

            logging.debug("receiving data")
            data = self.receive_all(rlist)

            if len(data):
                logging.debug("processing...")
                self.process_data(data)

            if msvcrt.kbhit() and msvcrt.getch() == b'e':
                print(f"current estimate of overall people seen is {self.estimate()}")

        print(f"current estimate of overall people seen is {self.estimate()}")
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
    lg_buckets = input("how many buckets would you like there to be in the counter? input in log 2 form")
    if not lg_buckets.isnumeric():
        logging.critical("inputted string must be an integer")
        return
    server = Server(lg_num_buckets=int(lg_buckets))
    server.run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(as"
                               ""
                               ""
                               "ctime)s: %(message)s [%(module)s, %(funcName)s(%(lineno)d)]",
                        datefmt="%H:%M:%S")
    main()
