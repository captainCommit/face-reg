import uuid

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