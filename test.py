import numpy as np
import cv2
import os
import urllib.request as urlreq

q
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_default.xml"

# check if file is in working directory
if haarcascade in os.listdir(os.curdir):
    print("Haarcascade model exists")
else:
    # download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
    urlreq.urlretrieve(haarcascade_url, haarcascade)
    print("Haarcascade model downloaded")

# Load the face detector
face_cascade = cv2.CascadeClassifier(haarcascade)


def normalize_face(cv2_img):
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)
    cv2.waitKey(0)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # If no faces are detected, return None
    if len(faces) == 0:
        return None

    # Assume only one face is detected, take the first one
    (x, y, w, h) = faces[0]

    # Calculate the center of the face
    cx = x + w // 2
    cy = y + h // 2

    # Calculate the angle of the face
    angle = 0

    # Calculate the desired location of the eyes in the output image
    desired_left_eye = (0.35, 0.35)
    desired_right_eye_x = 1.0 - desired_left_eye[0]
    desired_right_eye_y = desired_left_eye[1]

    # Calculate the scale of the output image
    desired_face_width = 256
    desired_face_height = 256
    scale = desired_face_width / w

    # Compute the transformation matrix
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    trans_x = desired_face_width * 0.5 - cx * scale
    trans_y = desired_face_height * desired_left_eye[1] - cy * scale
    rot_mat[0, 2] += trans_x
    rot_mat[1, 2] += trans_y

    # Apply the transformation to the input image
    output = cv2.warpAffine(cv2_img, rot_mat, (desired_face_width, desired_face_height))

    # Show the output image on the screen
    cv2.imshow("Output", output)

    return output


def main():
    cam = cv2.VideoCapture(0)

    while True:
        _, frame = cam.read()
        cv2.imshow("frame", frame)

        normalize_face(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
