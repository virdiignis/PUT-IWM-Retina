import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


class RetinaSegmentator:
    def __init__(self, path):
        self.image = plt.imread(path)

    def classical_deconstruction(self):
        image = self.image
        YCrCb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        y, cr, cb = cv.split(YCrCb)
        image = (image[::, ::, 1] * 2.1) - (image[::, ::, 1] * 1.9)
        image = image * 255 / image.max()
        image[image < 10] = 255

        # cr = cr * 255 / cr.max()
        # y = cr * 255 / cr.max()

        # image = image + cr + y
        print(image.max())

        # image = image - 20
        # image = np.clip(image, 0, 255)
        # image = np.clip(image*3, 0, 255)

        # black = np.zeros(image.shape)
        # black[image < 175] = 1


        # plt.imshow(image, cmap='gray')
        # plt.show()
        # return

        image = image.astype('uint8')

        kernel = np.ones((6, 6), np.uint8)
        image = cv.erode(image, kernel)
        kernel = np.ones((3, 3), np.uint8)
        image = cv.dilate(image, kernel)

        image = cv.bilateralFilter(image, 5, 210, 30)
        image = cv.medianBlur(image, 9)
        image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 71, 3)  # 61, 4
        image = cv.bilateralFilter(image, 5, 70, 10)
        image = cv.medianBlur(image, 9)

        # plt.imshow(image, cmap='gray')
        # plt.show()
        # return

        im = np.array(self.image)
        im[image > 127] = [0, 255, 255]
        plt.imshow(im, cmap='gray')
        plt.show()


if __name__ == '__main__':
    r = RetinaSegmentator('images/01_h.jpg')
    r.classical_deconstruction()
