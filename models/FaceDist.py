from utils import constants
from models.FaceDetector import FaceDetector

class FaceDist:
    """
    A class that creates a face map based on a list of images.

    Attributes:
        face_map (dict): A dictionary that maps face IDs to the images containing those faces.
        images (list): A list of image paths.
        face_detector (FaceDetector): An instance of the FaceDetector class.

    Methods:
        create_face_map(): Creates a face map based on the images provided.
    """

    def __init__(self, images, face_detector):
        self.face_map = {}
        self.images = images
        self.face_detector = face_detector if face_detector is not None else FaceDetector(mode = constants.FACE_DETECTION_MODES.OPEN_CV)

    def create_face_map(self):
        """
        Creates a face map based on the images provided.

        Returns:
            None
        """
        for image_path in self.images:
            self.face_detector.detect_faces(image_path)
            faces = self.face_detector.faces
            for face in faces:
                if face in self.face_map:
                    self.face_map[face.id]['images'].append(image_path)
                else:
                    self.face_map[face.id] = {
                        'face' : face,
                        'images' : [image_path]
                    }