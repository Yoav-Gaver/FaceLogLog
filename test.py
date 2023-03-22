import numpy as np
import cv2
import os
import urllib.request as urlreq
import dlib
import matplotlib.pyplot as plt
from deepface.commons import functions


haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml"
# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_alt.xml"

# save facial landmark detection model's name as LBFmodel
LBFmodel = "lbfmodel.yaml"


# check if file is in working directory
if haarcascade in os.listdir(os.curdir):
    print("Haarcascade model exists")
else:
    # download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
    urlreq.urlretrieve(haarcascade_url, haarcascade)
    print("Haarcascade model downloaded")

# Load the face detector
face_cascade = cv2.CascadeClassifier(haarcascade)

landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)


# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Define desired location of left eye in output image
desired_left_eye = (0.35, 0.35)


def normalize_face(cv2_img):
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # If no faces are detected, return None
    if len(faces) == 0:
        return None

    # Assume only one face is detected, take the first one
    (x, y, w, h) = faces[0]

    # Calculate the center of the face
    cx = int(x + w // 2)
    cy = int(y + h // 2)

    cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 5)
    cv2.circle(gray, (cx, cy), 5, (255, 0, 0), 3)
    cv2.imshow("Gray", gray)
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
    trans_x = desired_face_width * 0.5 - cx
    trans_y = desired_face_height * desired_left_eye[1] - cy
    rot_mat[0, 2] += trans_x
    rot_mat[1, 2] += trans_y

    # Apply the transformation to the input image
    output = cv2.warpAffine(
        cv2_img,
        rot_mat,
        (desired_face_width, desired_face_height))

    # Show the output image on the screen
    cv2.imshow("Output", output)

    return output


def align_face(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray, 1)

    # If no faces are detected, return None
    if len(faces) == 0:
        return None

    # Assume only one face is present
    face = faces[0]

    # Detect landmarks in the grayscale image
    landmarks = predictor(gray, face)

    # Calculate position of left and right eyes
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)

    # Calculate angle between eyes
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = -1 * (180 / np.pi) * np.arctan2(dy, dx)

    # Calculate scale factor based on distance between eyes
    distance = np.sqrt(dx*dx + dy*dy)
    desired_distance = (1.0 - desired_left_eye[0])
    scale = desired_distance * image.shape[1] / distance

    # Calculate rotation and translation matrices
    rot_mat = cv2.getRotationMatrix2D(left_eye, angle, scale)
    trans_mat = np.eye(3, dtype=np.float32)
    trans_mat[0, 2] = (image.shape[1] / 2) - left_eye[0]
    trans_mat[1, 2] = (image.shape[0] * desired_left_eye[1]) - left_eye[1]

    # Multiply rotation and translation matrices
    affine_mat = rot_mat.dot(trans_mat)

    # Apply transformation to image
    output_image = cv2.warpAffine(image, affine_mat[:2], (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    cv2.imshow('Output', output_image)
    return output_image


def align_and_crop_faces(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray, 1)
    cropped_faces = []
    # Loop through all detected faces
    for face in faces:
        # Detect landmarks in the grayscale image
        landmarks = predictor(gray, face)

        # Calculate position of left and right eyes
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)

        # Calculate angle between eyes
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = -1 * (180 / np.pi) * np.arctan2(dy, dx)

        # Calculate scale factor based on distance between eyes
        distance = np.sqrt(dx*dx + dy*dy)
        desired_distance = (1.0 - desired_left_eye[0])
        scale = desired_distance * face.shape[1] / distance

        # Calculate rotation and translation matrices
        rot_mat = cv2.getRotationMatrix2D(left_eye, angle, scale)
        trans_mat = np.eye(2, 3, dtype=np.float32)
        trans_mat[0, 2] = (face.shape[1] / 2) - left_eye[0]
        trans_mat[1, 2] = (face.shape[0] * desired_left_eye[1]) - left_eye[1]

        # Apply transformation to face
        aligned_face = cv2.warpAffine(face, rot_mat.dot(trans_mat), (face.shape[1], face.shape[0]), flags=cv2.INTER_LINEAR)

        # Crop face to include only the region inside the circle
        radius = int(distance / 2)
        center = (int((left_eye[0] + right_eye[0]) / 2), int((left_eye[1] + right_eye[1]) / 2))
        cropped_face = aligned_face[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius]

        # Draw circle around face and eyes in original image
        cv2.circle(image, center, radius, (0, 255, 0), 2)
        cv2.circle(image, left_eye, 2, (0, 0, 255), 2)
        cv2.circle(image, right_eye, 2, (0, 0, 255), 2)

        cropped_faces.append(cropped_face)

    # Show image with circles
    cv2.imshow("Faces with circles", image)

    # Return list of cropped faces
    return cropped_faces


def face_landmarks(image):
    image_cropped = image.copy()
    image_template = image_cropped.copy()
    image_gray = cv2.cvtColor(image_template, cv2.COLOR_RGB2GRAY)

    cv2.imshow("Gray", image_gray)

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = face_cascade.detectMultiScale(image_gray)

    # # Print coordinates of detected faces
    # print("Faces:\n", faces)

    for face in faces:
        #     save the coordinates in x, y, w, d variables
        (x, y, w, d) = face
        # Draw a white coloured rectangle around each face using the face's coordinates
        # on the "image_template" with the thickness of 2
        cv2.rectangle(image_template, (x, y), (x + w, y + d), (255, 255, 255), 2)

    cv2.imshow("Gray", image_gray)

    cv2.imshow("faces", image_template)

    # Detect landmarks on "image_gray"
    if len(faces) > 0:
        _, landmarks = landmark_detector.fit(image_gray, faces)

        # put landmarks on the image an
        landmarking(image_cropped, landmarks)

        cv2.imshow("landmarks", image_cropped)


def landmarking(image_cropped, landmarks):
    for landmark in landmarks:
        for x, y in landmark[0]:
            # display landmarks on "image_cropped"
            # with white colour in BGR and thickness 1
            cv2.circle(image_cropped, (int(x), int(y)), 1, (255, 255, 255), 1)


def main():
    cam = cv2.VideoCapture(0)

    while True:
        _, frame = cam.read()
        cv2.imshow("frame", frame)

        face_landmarks(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
