import numpy as np
import socket
import logging

PORT = 5440

MAX_LENLEN = 20
RECV_INTERVAL = 4096


def send(sock, vector):
    """
    Sends a message over the given socket.

    Args:
        sock (socket.socket):
        vector (np.ndarray):
        pubkey (rsa.key.PublicKey):
    """
    message = vector.tobytes()  # turn vector into bytes
    length = str(len(message))  # get length of bytes
    sock.send(length.zfill(MAX_LENLEN).encode())  # send the length
    sock.send(message)  # send the vector


def exit_socket(sock):
    message = b"exit"  # set message to exit
    length = str(len(message))  # get length of bytes
    sock.send(length.zfill(MAX_LENLEN).encode())  # send the length
    sock.send(message)


def recv(sock):
    """
    Receives a message over the given socket.

    Args:
        sock (socket.socket):
        privkey (rsa.key.PrivateKey):

    Returns:
        the decodes vector
    """
    try:
        length_bytes = sock.recv(MAX_LENLEN)  # get length of vector
    except:
        return None
    length = int(length_bytes.decode())  # decode length of vector
    message = b""
    while length > 0:
        recv_len = RECV_INTERVAL if length > RECV_INTERVAL else length
        message += sock.recv(recv_len)  # get message
        length -= recv_len

    if message == b"exit":
        return None

    vector = np.frombuffer(message, dtype=int)  # turn bytes to vector

    return vector
