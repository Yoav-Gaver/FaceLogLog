import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    """this class is for creating an object that can mesh faces in an image"""

    def __init__(self, static_mode=False, max_faces: int = 2, min_detection_con: int = 0.5, min_traking_con: int = 5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.min_tracking_con = min_traking_con
        self.min_detection_con = min_detection_con

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=self.static_mode, max_num_faces=self.max_faces,
                                                    min_tracking_confidence=self.min_traking_con,
                                                    min_detection_confidence=self.min_detection_con)
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=2)

    def FindFaceMesh(self, img, draw=True):
        """
        this method is used to draw and return the landmarks of the face

        :param img: the img to be processed
        :param draw: weather to draw the landmarks
        :rtype: list of lists of all the faces' landmarks
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(imgRGB)
        faces = []

        if results.multi_face_landmarks:
            for face_lms in results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, face_lms, self.mp_face_mesh.FACEMESH_CONTOURS,
                                                self.draw_spec, self.draw_spec)
                face = []
                for id, lm in enumerate(face_lms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    #             0.7, (255, 0, 0), 1)

                    face.append(([x, y]))
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(min_detection_con=0.7, maxfacese=3)

    ptime = time.time()

    while True:
        success, img = cap.read()
        img, faces = detector.FindFaceMesh(img, True)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)

        cv2.imshow("frame", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
