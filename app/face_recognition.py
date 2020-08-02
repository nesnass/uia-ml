import os
import os.path
import cv2

def detect_num_faces(original_file_path):
    datafilepath = os.path.join(os.getcwd(), 'app', 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(datafilepath)
    img = cv2.imread(original_file_path, cv2.IMREAD_COLOR)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.1, 4)
    return len(faces)
