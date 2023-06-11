import os
import cv2
from tqdm import tqdm
import face_recognition as faceReg

def get_face_encodings(image_path):
    img = faceReg.load_image_file(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = faceReg.face_locations(img_rgb)
    face_encodings = faceReg.face_encodings(img_rgb, face_locations)
    
    return face_encodings

def count_unique_faces(dir, image_paths, tolerance=0.6):
    known_face_encodings = []
    unique_faces_count = 0

    for image_path in tqdm(image_paths, ascii=True):
        face_encodings = get_face_encodings(f"{dir}\\{image_path}")
        for face_encoding in face_encodings:
            matches = faceReg.compare_faces(known_face_encodings, face_encoding, tolerance)
            if True not in matches:
                unique_faces_count += 1
                known_face_encodings.append(face_encoding)

    return unique_faces_count

def main():
    image_paths = os.listdir(dir:="images\\10")
    

    unique_faces = count_unique_faces(dir,image_paths, 0.4)
    print(f"Number of unique faces in the images: {unique_faces}")

if __name__ == "__main__":
    main()

