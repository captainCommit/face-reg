import os
import csv

from utils import constants
from models.FaceDetector import FaceDetector
from models.FaceDist import FaceDist
from models.FaceExtractor import FaceExtractor

class Orchestrator:
    """
    A class that orchestrates the face detection and extraction process.

    Attributes:
        folder (str): The path to the folder containing the images.

    Methods:
        run(): Runs the face detection and extraction process.
    """

    def __init__(self, folder, image_format = constants.IMAGE_FORMATS.JPEG, detection_mode = constants.FACE_DETECTION_MODES.OPEN_CV):
        self.folder = folder
        self.faceDector = FaceDetector(detection_mode)
        self.image_format = image_format 
        self.detection_mode = detection_mode

    def run(self):
        """
        Runs the face detection and extraction process.

        Returns:
            None
        """
        # Get the list of image files in the folder
        image_files = [os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith(f'.{self.image_format.value}')] 
        
        # Create a FaceMap instance

        face_map = FaceDist(image_files, self.faceDector)
        face_map.create_face_map()

        # Create a CSV file to store the face id and extracted face image path
        csv_file = 'face_data.csv'
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Face ID', 'Image Path'])

            # Extract and save faces from each image
            for face_data in face_map.face_map.values():
                images = face_data['images']
                face = face_data['face']
                for image in images:
                    face_extractor = FaceExtractor(image, face)
                    output_path = os.path.join(self.folder, 'extracted_faces')
                    os.makedirs(output_path, exist_ok=True)
                    save_path = face_extractor.extract_and_save_face(output_path)
                    writer.writerow([str(face.id), save_path])
        print('Face detection and extraction completed. CSV file created.')