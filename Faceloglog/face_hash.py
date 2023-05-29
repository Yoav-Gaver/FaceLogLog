import cv2
import numpy as np
import face_recognition
import time
import os

features_fix = np.array(
    [-0.09751934558153152, 0.09638582170009613, 0.05810072273015976, -0.05779742822051048, -0.11873337626457214,
     -0.005599715746939182, -0.022959548979997635, -0.09320846945047379, 0.1596250981092453, -0.07816176116466522,
     0.18420560657978058, -0.028074711561203003, -0.25986653566360474, -0.01801273226737976, -0.025314340367913246,
     0.12747067213058472, -0.15477746725082397, -0.13363765180110931, -0.10248526185750961, -0.07834549248218536,
     0.014461962506175041, 0.049281831830739975, 0.02084258571267128, 0.044966015964746475, -0.13677185773849487,
     -0.29959869384765625, -0.0673234611749649, -0.07804328203201294, 0.03678949922323227, -0.09074929356575012,
     0.009226702153682709, 0.06394369900226593, -0.18201705813407898, -0.047161005437374115, 0.030562829226255417,
     0.08344066143035889, -0.06635797768831253, -0.09704285115003586, 0.19508108496665955, 0.015524136833846569,
     -0.17471642792224884, -0.0062050893902778625, 0.06770472228527069, 0.2565787136554718, 0.2124362587928772,
     -0.003077669069170952, 0.02072245255112648, -0.07217143476009369, 0.13237981498241425, -0.27551478147506714,
     0.05505204573273659, 0.16014188528060913, 0.09853963553905487, 0.08011013269424438, 0.0836246907711029,
     -0.16234762966632843, 0.020880678668618202, 0.15228068828582764, -0.19399645924568176, 0.06583723425865173,
     0.07081516087055206, -0.08993736654520035, -0.028406815603375435, -0.06040892004966736, 0.1793772131204605,
     0.09937931597232819, -0.11167207360267639, -0.1470334380865097, 0.16366131603717804, -0.1549014300107956,
     -0.050797782838344574, 0.09854523837566376, -0.11000070720911026, -0.17735637724399567, -0.2633906900882721,
     0.039196670055389404, 0.3788008391857147, 0.14234115183353424, -0.15068373084068298, 0.02149500697851181,
     -0.07114896923303604, -0.024810954928398132, 0.01947341486811638, 0.06835068762302399, -0.06363566219806671,
     -0.046699151396751404, -0.09794549643993378, 0.032587192952632904, 0.20646101236343384, -0.019260132685303688,
     -0.025066545233130455, 0.23719894886016846, 0.015143269672989845, -0.01767364889383316, 0.029348600655794144,
     0.06478817760944366, -0.12359684705734253, -0.01659328117966652, -0.1209937110543251, -0.02024172432720661,
     0.028328891843557358, -0.09714996814727783, -0.0041018337942659855, 0.09646710008382797, -0.17543074488639832,
     0.16283375024795532, -0.020086126402020454, -0.014368360862135887, -0.02275494858622551, -0.047755349427461624,
     -0.08522437512874603, -0.0009511106181889772, 0.17113402485847473, -0.26657071709632874, 0.194911390542984,
     0.156102254986763, 0.016840791329741478, 0.13849034905433655, 0.054719217121601105, 0.06392785906791687,
     0.0036133285611867905, -0.0367475263774395, -0.1335757076740265, -0.0971313938498497, 0.03243446350097656,
     -0.03219699487090111, 0.02559298276901245, 0.039289142936468124])


class FaceHasher:
    def __init__(self, model: str = "models/opencv/haarcascade_frontalface_alt2.xml",
                 face_size: tuple[int] = (256, 256), show_images: bool = False):
        """
        creates the face hasher objects
        :param model: the path to the model to be used
        :param face_size: the size the face will be cut to
        :param show_images: whether to show the image
        """
        # Load the face detector
        self.face_cascade = cv2.CascadeClassifier(model)
        self.face_size = face_size
        self.show_images = show_images

    # TODO: add max faces to scan per frame
    def get_faces_vectors(self, frame: np.ndarray, do_round: bool = True):
        """
        create and return a vector for each face in the frame

        :param frame: image containing faces
        :param do_round: round the vector using a median


        :return: vectors representing the faces
        """
        # detect faces and crops for better time on later heavier methods
        faces_cord = self.detect_faces(frame)
        faces = self.crop_faces(frame, faces_cord)

        vectors = []
        # get vector of all faces
        for ind, face in enumerate(faces):
            v = self.get_face_vector(face, ind + 1,do_round=do_round)
            if v is not None:
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

    def get_face_vector(self, face: np.ndarray, ind: int = 0, do_round: bool = True):
        """
        Displays the input image and extracts features from the face using face recognition.

        :param face: The input image containing the face.
        :param ind: The index of the input image. Default is 0.
        :param do_round: round the vector using a median

        :return: the rounded vector representing the face
        """
        if self.show_images:
            cv2.imshow(f"face num {ind}", face)

        # get vector of face features
        features, face_locations = self.extract_features(face)

        if face_locations is None:
            return None

        rounded_features = features
        if type(features) == np.ndarray and do_round:
            rounded_features = np.ceil(np.subtract(features, features_fix)).astype(int)

        return rounded_features

    @staticmethod
    def leading_zeros(vector: np.ndarray, end_padding: int = 0) -> int:
        """
        Counts the number of leading zeros in a numpy array.

        :param vector: a numpy array
        :param end_padding: how many of the end is not used

        Returns:
        an integer representing the number of leading zeros in the array.
        """
        i = 0
        while vector[i] == 0 and i < len(vector) - end_padding:
            i += 1

        return i

    @staticmethod
    def extract_features(image: np.ndarray) -> (np.ndarray[float], list[list[int]]):
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


def get_features_mean(images_dir: str) -> list:
    """
    get the median vector of faces from a folder full of pictures

    :param images_dir: directery location
    """
    features = []
    face_hasher = FaceHasher(show_images=False)

    for path in os.listdir(images_dir):
        frame = cv2.imread(f"{images_dir}{path}")
        vectors = face_hasher.get_faces_vectors(frame=frame, do_round=False)

        cv2.imshow("frame", frame)

        for v in vectors:
            features.append(v)

        print("next frame")
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    median = []
    for ind in range(len(features[0])):
        l = []
        for feature in features:
            l.append(feature[ind])
        median.append(sorted(l)[len(l) // 2])

    print(median)


def main():
    cam = cv2.VideoCapture(0)
    face_vectorized = FaceHasher()

    ptime = time.time()

    max_zeros = 0
    while True:
        _, frame = cam.read()

        if len(frame) == 0:
            print("image empty check source")
            continue

        # detect faces and crops
        vectors = face_vectorized.get_faces_vectors(frame=frame)

        for ind, v in enumerate(vectors):
            print(f"face num {v + 1}: \n{v}")

        # add fps counter
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, f"FPS: {format(fps, '.3f')}", (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)

        # show frame
        cv2.imshow("frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cam.release()


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
