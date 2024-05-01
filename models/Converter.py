import os

import traceback
import rawpy
import imageio
from PIL import Image

from utils import constants


class Converter:
    """
    A class that converts images in a folder to a specified format.

    Attributes:
        folder (str): The path to the folder containing the images.
        output_format (str): The desired output format for the images.

    Methods:
        convert_images(): Converts the images in the folder to the specified format.
    """

    def __init__(self, folder, output_format = constants.IMAGE_FORMATS.JPEG):
        self.folder = folder
        self.output_format = output_format
    
    def convert_nef_to_jpg(self, in_path, out_path):
        with rawpy.imread(in_path) as raw:
            rgb = raw.postprocess()
            imageio.imwrite(out_path, rgb)

    def convert_images(self):
        """
        Converts the images in the folder to the specified format.

        Returns:
            None
        """
        image_files = [file for file in os.listdir(self.folder) if file.lower().endswith((constants.IMAGE_FORMATS.NEF.value, constants.IMAGE_FORMATS.JPG.value))]

        for image_file in image_files:
            image_path = os.path.join(self.folder, image_file)
            output_folder = os.path.join(self.folder,'converted-images')
            os.makedirs(output_folder, exist_ok=True)
            try:
                if image_file.lower().endswith(constants.IMAGE_FORMATS.NEF.value):
                    self.convert_nef_to_jpg(image_path, os.path.join(output_folder,os.path.splitext(image_file)[0]+f".{self.output_format.lower()}"))
                    print(f"Converted {image_file} to {self.output_format.upper()}")
                    continue
                image = Image.open(image_path)
                new_width  = 800
                new_height = new_width * image.size[1] // image.size[0]
                image = image.resize((new_width, new_height), Image.LANCZOS)
                image.save(os.path.join(output_folder,os.path.splitext(image_file)[0]+"."+self.output_format.lower()), self.output_format.upper())
                print(f"Converted {image_file} to {self.output_format.upper()}")
            except Exception as e:
                print(traceback.format_exc())
                print(f"Failed to convert {image_file}: {str(e)}")