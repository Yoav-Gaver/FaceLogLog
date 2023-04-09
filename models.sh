mkdir models
cd models
mkdir dlib openface

# download and extract the shape predictor model
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# download the OpenFace model
wget https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7 -O openface/nn4.small2.v1.t7
