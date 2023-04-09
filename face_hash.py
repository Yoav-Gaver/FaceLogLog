import cv2
import mediapipe as mp
import time
import numpy as np
import logging

i = 0


class FaceMeshDetector:
    """this class is for creating an object that can mesh faces in an image"""

    def __init__(self, static_mode=False, max_faces: int = 2, min_detection_con: float = 0.5,
                 min_tracking_con: float = 0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.min_tracking_con = min_tracking_con
        self.min_detection_con = min_detection_con

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=self.static_mode, max_num_faces=self.max_faces,
                                                    min_tracking_confidence=self.min_tracking_con,
                                                    min_detection_confidence=self.min_detection_con)
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=2)

    def FindFaceMesh(self, img, draw=True):
        """
        this method is used to draw and return the landmarks of the face

        :param img: the img to be processed
        :param draw: weather to draw the landmarks
        :rtype: list of lists of all the faces' landmarks
        """
        global i

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(imgRGB)
        faces = []

        if results.multi_face_landmarks:
            for face_lms in results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        image=img, landmark_list=face_lms,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.draw_spec,
                        connection_drawing_spec=self.draw_spec)
                face = []
                # for ind, lm in enumerate(face_lms.landmark):
                self.put_landmark(face_lms, i, img, True)
                self.put_landmark(face_lms, 133, img)
                self.put_landmark(face_lms, 362, img)
                self.put_landmark(face_lms, 54, img)
                self.put_landmark(face_lms, 284, img)
                print(self.landmark_dist(face_lms.landmark[284], face_lms.landmark[54]) /
                      self.landmark_dist(face_lms.landmark[133], face_lms.landmark[362]))
                # print(self.landmark_dist(face_lms, 154, 362))

                # face.append(([x, y]))
                # faces.append(face)
        return img, faces

    def put_landmark(self, face_lms, ind, img, draw_number=False):
        lm = face_lms.landmark[ind]
        ih, iw, _ = img.shape

        x, y = int(lm.x * iw), int(lm.y * ih)
        if draw_number:
            cv2.putText(img, str(ind), (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 0, 0), 1)
        else:
            cv2.circle(img, (x, y),
                       self.draw_spec.circle_radius,
                       (255, 0, 0),
                       self.draw_spec.thickness)


    def landmark_dist(self, lm1, lm2):
        lm1_point = np.array((lm1.x, lm1.y, lm1.z))
        lm2_point = np.array((lm2.x, lm2.y, lm2.z))

        return np.sum(np.square(lm1_point - lm2_point))


def main():
    global i
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(min_detection_con=0.9, static_mode=False)

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
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('w'):
            i += 1
        if key == ord('e'):
            i += 10
        if key == ord('s'):
            i -= 1
        if key == ord('d'):
            i -= 10


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(relativeCreated)d|%(filename)s:: %(funcName)s: %(msg)s")
    main()
