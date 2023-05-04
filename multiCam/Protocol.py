import numpy as np
import socket
import rsa


def send_key(sock, pubkey):
    """Sends a public key over the given socket.

    Args:
        sock (socket.socket):
        pubkey (rsa.key.PublicKey):
    """

    key_bytes = pubkey.save_pkcs1()                          # turn key into bytes
    sock.send(len(key_bytes).to_bytes(14, byteorder='big'))  # put key in format
    sock.send(key_bytes)                                     # send key


def recv_key(sock):
    """Receives a public key over the given socket.

    Args:
        sock (socket.socket):

    Returns:
        the public key for the rsa encryption.
    """
    key_len_bytes = sock.recv(14)                                    # get length of key
    key_len       = int.from_bytes(key_len_bytes, byteorder='big')   # decode length
    key_bytes     = sock.recv(key_len)                               # get key
    pubkey        = rsa.PublicKey.load_pkcs1(key_bytes)              # decode key
    return pubkey


def send(sock, vector, pubkey):
    """Sends an RSA-encrypted message over the given socket.

    Args:
        sock (socket.socket):
        vector (np.ndarray):
        pubkey (rsa.key.PublicKey):
    """
    message   = np.array2string(vector)                # turn vector into bytes
    encrypted = rsa.encrypt(message.encode(), pubkey)  # decode bytes using key
    length    = len(encrypted)                         # get length of bytes
    sock.send(length.to_bytes(4, byteorder='big'))     # send the length
    sock.send(encrypted)                               # send the vector


def exit_socket(sock, pubkey):
    message   = "exit"                                 # set message to exit
    encrypted = rsa.encrypt(message.encode(), pubkey)  # decode bytes using key
    length    = len(encrypted)                         # get length of bytes
    sock.send(length.to_bytes(4, byteorder='big'))     # send the length
    sock.send(encrypted)


def recv(sock, privkey):
    """Receives an RSA-encrypted message over the given socket and decrypts it.

    Args:
        sock (socket.socket):
        privkey (rsa.key.PrivateKey):

    Returns:
        the decodes vector
    """
    length_bytes = sock.recv(4)                         # get length of vector
    length       = int.from_bytes(length_bytes,
                                  byteorder='big')      # decode length of vector
    encrypted    = sock.recv(length)                    # get encrypted vector
    decrypted    = rsa.decrypt(encrypted, privkey)      # decrypt vector
    if decrypted == b'exit':                            # check if
        return None
    vector       = np.frombuffer(decrypted, dtype=int)  # turn bytes to vector

    return vector
