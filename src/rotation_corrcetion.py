
@dataclasses
class StatusRotation():
    """
    This object contains the list of the templates image and the list of the image
    """
    templates = [] #lista di immagini"



class RotationCorrection():

    def load_image(self,filepath: str):
        """
        This function loads an image from a fime

        Args:
            filepath: string

        Returns:
            image: image loaded
        """

    def correct_image(self,image, x: float = 0, y: float = 0, angle: float = 0, shift_first: bool = True, y_bottom_up: bool =True):
        """
        This function copies the input image in variable image_out and applies a x and y shift and a rotation to the image_out.

        Args:
            image: image input
            x: shift in pixel. positive left to right
            y: shift in pixel. positive bottom to up
            angle: rotation angle in degrees. Positive CCW
            shift_first: boolean which determines whether to do the shift or the rotation first
            y_bottom_up: boolean which determines the direction of the y shift. If "False", then positive direction is up to bottom

        Returns:
            image_out: image corrected by shift and rotation
        """

    def
