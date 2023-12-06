from deepface import DeepFace
from retinaface import RetinaFace
import os
import cv2
import matplotlib.pyplot as plt
import logging
import traceback

class FaceRecognition:

    def verify(self, img1_path, img2_path):

        try:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            plt.imshow(img1[:, :, ::-1])
            plt.show()
            plt.imshow(img2[:, :, ::-1])
            plt.show()

            output = DeepFace.verify(img1_path, img2_path, enforce_detection=False)
            print(output)
            verification = output['verified']
            if verification:
                print("same pictures!")
            else:
                print("different pictures!")
        except:
            logging.error(traceback.format_exc())

    def image_information(self, image_path):
        face = RetinaFace.extract_faces(image_path)
        result = DeepFace.analyze(face, detector_backend='skip')
        print(result)

first_img = "/home/mohammadjavad/Desktop/face/10.jpg"
second_img = "/home/mohammadjavad/Desktop/face/4.jpg"
thired_img = "/home/mohammadjavad/Desktop/face/6.jpg"
face_recog=FaceRecognition()
face_recog.verify(first_img, thired_img)
face_recog.image_information(thired_img)