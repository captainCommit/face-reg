import uuid
import cv2
import os
import csv

class FaceDetector:
    """
    A class that detects faces in an image using Haar cascades.

    Attributes:
        faces (list): A list to store the detected faces.
        face_cascade (cv2.CascadeClassifier): The Haar cascade classifier for face detection.

    Methods:
        detect_faces(image_path): Detects faces in the given image and stores them in the 'faces' attribute.
    """

    def __init__(self):
        self.faces = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, image_path):
        """
        Detects faces in the given image and stores them in the 'faces' attribute.

        Args:
            image_path (str): The path to the image file.

        Returns:
            None
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        self.faces = [Face((x, y, w, h), None, None) for (x, y, w, h) in faces]


class Face:
    """
    A class that represents a face detected in an image.

    Attributes:
        bounding_box (tuple): The bounding box coordinates of the face.
        name (str): The name associated with the face.
        contact_method (str): The contact method associated with the face.
        id (uuid.UUID): The unique identifier for the face.

    Methods:
        set_name(name): Sets the name of the face.
        set_contact_method(contact_method): Sets the contact method of the face.
    """

    def __init__(self, bounding_box, name, contact_method):
        self.bounding_box = bounding_box
        self.name = name
        self.contact_method = contact_method
        self.id = uuid.uuid4()

    def set_name(self, name):
        """
        Sets the name of the face.

        Args:
            name (str): The name to be set.

        Returns:
            None
        """
        self.name = name

    def set_contact_method(self, contact_method):
        """
        Sets the contact method of the face.

        Args:
            contact_method (str): The contact method to be set.

        Returns:
            None
        """
        self.contact_method = contact_method


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
        self.face_detector = face_detector if face_detector is not None else FaceDetector()

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


class FaceExtractor:
    """
    A class that extracts and saves the face from an image.

    Attributes:
        image (str): The path to the image file.
        face (Face): The face object containing the bounding box coordinates.

    Methods:
        extract_and_save_face(output_path): Extracts and saves the face from the image.
    """

    def __init__(self, image, face):
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

class Orchestrator:
    """
    A class that orchestrates the face detection and extraction process.

    Attributes:
        folder (str): The path to the folder containing the images.

    Methods:
        run(): Runs the face detection and extraction process.
    """

    def __init__(self, folder):
        self.folder = folder
        self.faceDector = FaceDetector()
    def run(self):
        """
        Runs the face detection and extraction process.

        Returns:
            None
        """
        # Get the list of image files in the folder
        image_files = [os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith('.jpg')]

        # Create a FaceMap instance
        face_map = FaceDist(image_files, self.face_detector)
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
                    face_extractor.extract_and_save_face(output_path)

                    writer.writerow([str(face.id), os.path.join(output_path, str(face.id) + '.jpg')])

                print('Face detection and extraction completed. CSV file created.')
                

