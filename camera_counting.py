import cv2
import os
import urllib.request as urlreq
import matplotlib.pyplot as plt

# save face detection algorithm's url in haarcascade_url variable
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_alt2.xml"

# save facial landmark detection model's url in LBFmodel_url variable
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

# save facial landmark detection model's name as LBFmodel
LBFmodel = "lbfmodel.yaml"


def initiate():
    # check if file is in working directory
    if haarcascade in os.listdir(os.curdir):
        print("Haarcascade model exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
        urlreq.urlretrieve(haarcascade_url, haarcascade)
        print("Haarcascade model downloaded")

    # check if file is in working directory
    if (LBFmodel in os.listdir(os.curdir)):
        print("LBF model exists")
    else:
        # download picture from url and save locally as lbfmodel.yaml, < 54MB
        urlreq.urlretrieve(LBFmodel_url, LBFmodel)
        print("LBF model downloaded")


def main():
    cam = cv2.VideoCapture(0)

    # create an instance of the Face Detection Cascade Classifier
    detector = cv2.CascadeClassifier(haarcascade)

    while True:
        _, frame = cam.read()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # convert image to RGB colour
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # set dimension for cropping image
        x, y, width, depth = 50, 200, 950, 500
        image_cropped = image_rgb[y:(y + depth), x:(x + width)]

        # create a copy of the cropped image to be used later
        image_template = image_cropped.copy()

        # convert image to Grayscale
        image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

        # show cropped and grayed image
        cv2.imshow("gray", image_gray)

        # Detect faces using the haarcascade classifier on the "grayscale image"
        faces = detector.detectMultiScale(image_gray)

        # Print coordinates of detected faces
        print("Faces:\n", faces)

        for face in faces:
            #     save the coordinates in x, y, w, d variables
            (x, y, w, d) = face
            # Draw a white coloured rectangle around each face using the face's coordinates
            # on the "image_template" with the thickness of 2
            cv2.rectangle(image_template, (x, y), (x + w, y + d), (0, 0, 255), 2)

        plt.axis("off")
        plt.imshow(image_template)
        plt.title('Face Detection')
        plt.pause(0.01)

    # After the loop release the cap object
    cam.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


initiate()
if __name__ == '__main__':
    main()

    