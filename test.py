import cv2
import numpy as np
from imutils.face_utils import FaceAligner, rect_to_bb
import dlib
import torch
import torchvision.transforms as transforms
import urllib.request as urlreq
import os
import facenet_pytorch
import face_recognition


FACE_SIZE = (256, 256)

haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
# save face detection algorithm's name as haarcascade
haarcascade = "models/opencv/haarcascade_frontalface_alt2.xml"

# save facial landmark detection model's name as LBFmodel
LBFmodel = "models/opencv/lbfmodel.yaml"


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


def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    return faces


def crop_faces(image, faces):
    face_images = []
    for (x, y, w, h) in faces:
        cropped = image[y:y+h, x:x+w]
        face_images.append(cv2.resize(cropped, FACE_SIZE))
    return face_images


def get_embeddings_and_faces(image):
    predictor_model = 'models/dlib/shape_predictor_68_face_landmarks.dat'
    face_recognition_model = 'models/openface/nn4.small2.v1.t7'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)
    face_recognizer = torch.load(face_recognition_model).eval()

    # define face aligner
    face_aligner = FaceAligner(predictor, desiredFaceWidth=96, desiredLeftEye=(0.3, 0.3))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)
    embeddings = []
    aligned_faces = []
    for rect in rects:
        # align face
        aligned_face = face_aligner.align(image, gray, rect)
        # convert aligned face to tensor
        aligned_face = transforms.functional.to_tensor(aligned_face)
        aligned_face = torch.unsqueeze(aligned_face, 0)
        # forward pass through face recognition model
        embedding = face_recognizer(aligned_face)
        embeddings.append(embedding.detach().numpy().flatten())
        aligned_faces.append(aligned_face.numpy().squeeze().transpose((1, 2, 0)))
    return np.array(embeddings), aligned_faces


def extract_features_with_facenet_pytorch(img):
    # Find all the faces in the image
    detector = facenet_pytorch.MTCNN()
    face_locations, _ = detector.detect(img)

    # Align and extract features for each face
    model = facenet_pytorch.InceptionResnetV1(pretrained='vggface2').eval()
    features = []
    for i, face_location in enumerate(face_locations):
        # Extract the face from the image
        top, right, bottom, left = face_location
        print(face_location)
        face_img = img[int(top):int(bottom), int(left):int(right)]

        # Align the face using landmarks
        landmarks = detector.detect_points(face_img)[0]
        aligned_face_img = facenet_pytorch.functional.align_face(face_img, landmarks)

        # Extract features using a pre-trained model
        aligned_face_tensor = facenet_pytorch.preprocess(aligned_face_img)
        with torch.no_grad():
            face_features = model(aligned_face_tensor.unsqueeze(0))[0]

        # Process the features (replace this with your own processing logic)
        processed_features = np.mean(face_features.numpy())

        # Append the processed features to the list
        features.append(processed_features)

    # Return the processed features and the face locations
    return features, face_locations


def extract_features_with_face_recognition(image):
    # Find all the faces in the image
    face_locations = face_recognition.face_locations(image)

    # Load the face recognition model to extract features
    model = face_recognition.face_encodings(image, face_locations)

    # Process the face features (replace this with your own processing logic)
    processed_features = np.mean(model, axis=0)

    # Return the processed features and the face locations
    return processed_features, face_locations


def main():
    cam = cv2.VideoCapture(0)

    if "images" not in os.listdir():
        os.mkdir("images")

    while True:
        _, frame = cam.read()
        cv2.imshow("frame", frame)
        # # detect faces
        # features, face_locations = extract_features_with_face_recognition(frame)
        # rounded_features = np.round(np.clip(features, -0.5, 0.5) / 0.2) * 0.2
        # print(rounded_features)
        # for ind, features in enumerate(features):
        #     # cv2.imshow(f"face num {ind}", face_img)
        #     print(f"face num {ind} features is: {features[ind]}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"images\\face{len(os.listdir('images'))}.jpg", frame)


if __name__ == '__main__':
    main()


