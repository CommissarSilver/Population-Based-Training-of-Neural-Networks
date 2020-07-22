import matplotlib.pyplot as plt
from skimage import data, io


def show_frame(frame):
    io.imshow(frame)
    plt.show()
