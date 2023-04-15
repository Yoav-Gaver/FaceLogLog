import cv2
import numpy as np
import face_recognition
import time

features_fix = np.array(
    [-0.11174381524324417, 0.07454776763916016, 0.05843915790319443, -0.07979366928339005, -0.10955216735601425,
     -0.02817821316421032, -0.006671702489256859, -0.14565499126911163, 0.19013269245624542, -0.13374821841716766,
     0.1816103309392929, -0.06185418367385864, -0.231658935546875, -0.04647332429885864, -0.04212888702750206,
     0.195501908659935, -0.16566839814186096, -0.19090986251831055, -0.06556224822998047, -0.04335155710577965,
     0.02328343689441681, 0.004496458917856216, 0.028715969994664192, 0.05193959176540375, -0.1321442872285843,
     -0.3570448160171509, -0.09374495595693588, -0.0924902856349945, 0.0081730792298913, -0.06112262234091759,
     -0.00015513133257627487, 0.07708559185266495, -0.1861874759197235, -0.0430966280400753, 0.02941509336233139,
     0.09729216992855072, -0.03915860131382942, -0.08941978216171265, 0.20368197560310364, -0.009334910660982132,
     -0.25246766209602356, -0.017495393753051758, 0.07160467654466629, 0.23868413269519806, 0.19593560695648193,
     -0.011811258271336555, 0.024091891944408417, -0.10467122495174408, 0.12035799771547318, -0.2498149424791336,
     0.01827254146337509, 0.13116663694381714, 0.09549511224031448, 0.059493452310562134, 0.05150827020406723,
     -0.14786064624786377, 0.02252870798110962, 0.12982232868671417, -0.20083215832710266, 0.028725331649184227,
     0.08197056502103806, -0.11470416188240051, -0.03249898925423622, -0.05487708002328873, 0.2055114060640335,
     0.1116328164935112, -0.10669885575771332, -0.1615544855594635, 0.18680573999881744, -0.1684175282716751,
     -0.030549151822924614, 0.0719098150730133, -0.1126958578824997, -0.1737293004989624, -0.2637648582458496,
     -0.005787527188658714, 0.38489824533462524, 0.12129265815019608, -0.14781920611858368, 0.029963236302137375,
     -0.08125844597816467, 0.023290134966373444, 0.022359861060976982, 0.08078223466873169, -0.04439682513475418,
     -0.022565647959709167, -0.10478255897760391, -0.001458178274333477, 0.2182334065437317, -0.054871443659067154,
     -0.003257553093135357, 0.21069328486919403, -0.005236098542809486, -0.005087297409772873, 0.016337525099515915,
     0.06796122342348099, -0.11619183421134949, 0.014581764116883278, -0.13654807209968567, -0.011841144412755966,
     -0.006817013956606388, -0.04847026243805885, 0.005097370594739914, 0.09713845700025558, -0.19076034426689148,
     0.1442538946866989, -0.02622356079518795, -0.0304985661059618, -0.01489903125911951, -0.017737988382577896,
     -0.05286156013607979, -0.03472384065389633, 0.14690174162387848, -0.23866881430149078, 0.17036263644695282,
     0.1824260950088501, 0.014471100643277168, 0.16053825616836548, 0.05338404327630997, 0.07387612760066986,
     -0.021614067256450653, -0.042450081557035446, -0.16080546379089355, -0.0703396126627922, 0.057660605758428574,
     -0.036869537085294724, 0.027016807347536087, 0.028477180749177933])


class face_hasher():
    def __int__(self, haarcascade="models/opencv/haarcascade_frontalface_alt2.xml", face_size=(256, 256)):
        """
        creates
        :param haarcascade:
        :param face_size:
        :return:
        """
        # Load the face detector
        self.face_cascade = cv2.CascadeClassifier(haarcascade)
        self.face_size = face_size

    #TODO: add max faces to scan per frame
    def get_faces_vectors(self, frame: np.ndarray):
        """
        create and return a vector for each face in the frame

        :param frame: image containing faces
        :return: vectors representing the faces
        """
        # detect faces and crops for better time on later heavier methods
        faces_cord = self.detect_faces(frame)
        faces = self.crop_faces(frame, faces_cord)

        vectors = []
        # get vector of all faces
        for ind, face in enumerate(faces):
            v = self.get_face_vector(face, ind + 1)
            if v != None:
                vectors.append(v)

        return vectors


    def detect_faces(self, image: np.ndarray) -> list[list[int]]:
        """
        gets an images that includes faces and returns their coordinates

        :param image: image of type numpy array
        :return: the coordinates of all faces in the image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        return faces

    def crop_faces(self, image: np.ndarray, faces: list[list[int]]) -> list[np.ndarray]:
        """
        gets an image and coordinates of faces and crops the image accordingly

        :param image: the image containing the faces
        :param faces: the coordinates of the faces
        :return: images of the cropped faces
        """
        face_images = []
        for (x, y, w, h) in faces:
            cropped = image[y:y + h, x:x + w]
            face_images.append(cv2.resize(cropped, self.face_size))
        return face_images

    def get_face_vector(self, face: np.ndarray, ind: int = 0):
        """
        Displays the input image and extracts features from the face using face recognition.

        :param face: The input image containing the face.
        :param ind: The index of the input image. Default is 0.

        :return: the rounded vector representing the face
        """
        cv2.imshow(f"face num {ind}", face)

        # get vector of face features
        features, face_locations = self.extract_features_with_face_recognition(face)

        if face_locations == None:
            return None
        if type(features) == np.ndarray:
            rounded_features = np.ceil(np.subtract(features, features_fix)).astype(int)
            print(f"face number {ind}: {rounded_features}")
        return rounded_features

    @staticmethod
    def leading_zeros(vector: np.ndarray) -> int:
        """
        Counts the number of leading zeros in a numpy array.

        :param vector: a numpy array

        Returns:
        an integer representing the number of leading zeros in the array.
        """
        i = 0
        while vector[i] == 0 and i < len(vector):
            i += 1

        return i

    @staticmethod
    def extract_features_with_face_recognition(image: np.ndarray) -> (np.ndarray[float], list[list[int]]):
        """
        extract a vector from an image of a face

        :param image: the image containing a face
        :return: a vector representing the face
        """
        # Find all the faces in the image
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) <= 0:
            return None, None

        # Load the face recognition model to extract features of all faces in the frame
        model = face_recognition.face_encodings(image, face_locations)

        # Process the face features (replace this with your own processing logic)
        processed_features = np.mean(model, axis=0)

        # Return the processed features and the face locations
        return processed_features, face_locations


def main():
    cam = cv2.VideoCapture(0)
    face_vectorized = face_hasher()

    ptime = time.time()

    max_zeros = 0
    while True:
        _, frame = cam.read()

        if len(frame) == 0:
            print("image empty check source")
            continue

        # detect faces and crops
        vectors = face_vectorized.get_faces_vectors(frame=frame)

        max_zero_in_frame = 0
        for v in vectors:
            zeros = face_hasher.leading_zeros(v)

            max_zero_in_frame = max(max_zero_in_frame, zeros)

        max_zeros = max(max_zero_in_frame, max_zeros)
        print(f"number of maximum leading zeros in frame: {max_zero_in_frame}\noverall: {max_zeros}")

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, f"FPS: {format(fps, '.3f')}", (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)

        cv2.imshow("frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
