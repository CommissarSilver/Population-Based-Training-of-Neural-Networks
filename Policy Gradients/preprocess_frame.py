from skimage import transform
import numpy as np


def preprocess_frame(frame):
    # Arguments:
        # frame: a single frame of the game outputted by VizDoom
    # Returns:
        # preprocessed_frame: a grayscale, 84*84 matrix
    # Implements:
        # A single frame is in RGB. Since we don't need color for analyzing what's going on in the game, we first
        # convert it to grayscale. After doing so we crop the roof and also the bottom information bar out of the frame
        # since they are also not needed for playing the game.
        # After that we normalize the cropped frame.
        # Finally by using skimage's transform module we shrink the image even further and return the resulting matrix.

    grayscale_frame = np.mean(frame, axis=0)  # convert from RGB to grayscale
    cropped_frame = grayscale_frame[30:-10, 30:-30]  # crop the roof and bottom info bar out
    normalized_frame = cropped_frame / 255  # normalize the cropped frame
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])  # shrink the cropped frame

    return preprocessed_frame
