import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import data, io
import numpy as np


def show_frame(frame):
    io.imshow(frame)
    plt.show()
