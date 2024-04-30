from enum import Enum
class IMAGE_FORMATS(str, Enum):
    JPG='jpg' 
    JPEG='jpeg'
    PNG='png'
    BMP='bmp'
    GIF='gif'
    TIFF='tiff'
    NEF='nef',
    WEBP='webp'
class CONTACT_METHODS(str, Enum):
        EMAIL='email'
        PHONE='phone'
        SOCIAL_MEDIA='social_media'
    
class FACE_DETECTION_MODES(str, Enum):
        OPEN_CV='open_cv'
        MTCNN="mtcnn"
    