from mtcnn import MTCNN
import cv2
import utils.constants as constants
from models.Face import Face

class FaceDetector:
    """
    A class that detects faces in an image using Haar cascades.

    Attributes:
        faces (list): A list to store the detected faces.
        face_cascade (cv2.CascadeClassifier): The Haar cascade classifier for face detection.

    Methods:
        detect_faces(image_path): Detects faces in the given image and stores them in the 'faces' attribute.
    """

    def __init__(self, mode = constants.FACE_DETECTION_MODES.MTCNN):
        self.faces = []
        self.mode = mode
        if mode == constants.FACE_DETECTION_MODES.OPEN_CV:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        elif mode == constants.FACE_DETECTION_MODES.MTCNN:
            self.face_cascade = MTCNN()
    def detect_faces(self, image_path):
        """
        Detects faces in the given image and stores them in the 'faces' attribute.

        Args:
            image_path (str): The path to the image file.

        Returns:
            None
        """
        image = cv2.imread(image_path)
        if self.mode == constants.FACE_DETECTION_MODES.OPEN_CV:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            self.faces = [Face((x, y, w, h), None, None) for (x, y, w, h) in faces]
        elif self.mode == constants.FACE_DETECTION_MODES.MTCNN:
            self.face_cascade = MTCNN()
            print(self.face_cascade.detect_faces(image))
            self.faces = [Face((face['box'][0],face['box'][1],face['box'][2],face['box'][3]), None, None) for face in self.face_cascade.detect_faces(image)]