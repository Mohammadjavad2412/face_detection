from deepface import DeepFace
from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt
import logging
import traceback


models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'fastmtcnn',
]
class FaceRecognition:

    def verify(self, img1_path, img2_path):

        try:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            plt.imshow(img1[:, :, ::-1])
            plt.show()
            plt.imshow(img2[:, :, ::-1])
            plt.show()

            output = DeepFace.verify(img1_path, img2_path, detector_backend=backends[3] ,model_name='DeepFace')
            print(output)
            verification = output['verified']
            if verification:
                print("same pictures!")
            else:
                print("different pictures!")
        except:
            logging.error(traceback.format_exc())

    def image_information(self, image_path, db_path):
        # result = DeepFace.analyze(image_path, enforce_detection=False)
        for i in range(len(models)):  
          findings = DeepFace.find(image_path, "/home/mj/Desktop/documents/3d_images_art", enforce_detection=False ,detector_backend=backends[9], model_name=models[i])
          print(models[i])
          print(findings)
        # print(result)


    def stream_face_detection(self, db_path):
        DeepFace.stream(db_path, model_name=models[0], frame_threshold=20,distance_metric='cosine', enable_face_analysis=False, detector_backend=backends[0])

first_img = "/home/mj/Desktop/documents/3d_images_art/4.jpg"
second_img = "/home/mj/Desktop/documents/3d_images_art/5.jpg"
thired_img = "/home/mj/Desktop/documents/3d_images_art/4.jpg"
db_path = "/home/mj/Desktop/documents/3d_images_art/general"
face_recog=FaceRecognition()
# face_recog.verify(first_img, second_img)
# face_recog.image_information(thired_img, db_path)
face_recog.stream_face_detection(db_path)