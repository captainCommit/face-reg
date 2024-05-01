import cv2
from models.Face import Face
class FaceExtractor:
    """
    A class that extracts and saves the face from an image.

    Attributes:
        image (str): The path to the image file.
        face (Face): The face object containing the bounding box coordinates.

    Methods:
        extract_and_save_face(output_path): Extracts and saves the face from the image.
    """

    def __init__(self, image, face : Face):
        self.image = image
        self.face = face

    def extract_and_save_face(self, output_path):
        """
        Extracts and saves the face from the image.

        Args:
            output_path (str): The path to save the extracted face.

        Returns:
            None
        """
        x, y, w, h = self.face.bounding_box
        face_image = cv2.imread(self.image)
        face_crop = face_image[y:y+h, x:x+w]

        if self.face.name:
            filename = self.face.name
        else:
            filename = str(self.face.id)

        save_path = output_path + '/' + filename + '.jpg'
        cv2.imwrite(save_path, face_crop)
        return save_path