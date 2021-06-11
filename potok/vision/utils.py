from autocrop import Cropper
import cv2


def crop_face(img, dim=256):
    # height, width, channels = img.shape
    # dim = min(height, width)
    
    cropper = Cropper(width=dim, height=dim)
    cropped = cropper.crop(img)
    if cropped is not None:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return cropped


