import matplotlib.pyplot as plt
from skimage.util import random_noise
import skimage.io as io

def noise_augmentation(img):  
    return random_noise(img,var=0.1)
