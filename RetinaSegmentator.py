import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


class RetinaSegmentator:
    def __init__(self, path):
        image = cv.imread(path)
        # plt.imshow(image)

        image = (image[::, ::, 1] * 2.1) - (image[::, ::, 1] * 1.9)
        image = image * 255 / image.max()
        image = image.astype('uint8')
        # image = cv.equalizeHist(image)

        kernel = np.ones((6, 6), np.uint8)
        image = cv.erode(image, kernel)
        kernel = np.ones((3, 3), np.uint8)
        image = cv.dilate(image, kernel)
        # plt.imshow(image, cmap='gray')
        # plt.show()
        # return

        image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 131, 4)

        image = cv.medianBlur(image, 5)
        # image = cv.bilateralFilter(image, 5, 1, 9)
        plt.imshow(image, cmap='gray')
        plt.show()


if __name__ == '__main__':
    r = RetinaSegmentator('images/01_h.jpg')
