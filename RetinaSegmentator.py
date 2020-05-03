from multiprocessing import Manager, Pool
from multiprocessing import Process

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from h2o4gpu.neural_network import MLPClassifier
from h2o4gpu.preprocessing import StandardScaler
from h2o4gpu.model_selection import train_test_split, cross_val_score
from joblib import dump, load


class RetinaSegmentator:
    def __init__(self):
        self._pool = Pool()

    @classmethod
    def classical_deconstruction(cls, image):
        orig_image = image
        image = cls._basic_processing(image)

        kernel = np.ones((6, 6), np.uint8)
        image = cv.erode(image, kernel)
        kernel = np.ones((3, 3), np.uint8)
        image = cv.dilate(image, kernel)

        image = cv.bilateralFilter(image, 5, 210, 30)
        image = cv.medianBlur(image, 9)
        image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 3)  # 61, 4
        image = cv.bilateralFilter(image, 5, 70, 10)
        image = cv.medianBlur(image, 9)

        im = np.array(orig_image)
        im[image > 50] = [0, 255, 255]
        plt.imshow(im, cmap='gray')
        plt.show()
        return image

    def reconstruct_and_compare(self, i):
        source = cv.imread(f"images/{i:02d}_h.jpg")
        mask = cv.imread(f"images/masked/{i:02d}_h.tif", cv.IMREAD_GRAYSCALE)

        rec = self.classical_deconstruction(source)
        self.show_difference(rec, mask)

    @staticmethod
    def show_difference(img1: np.ndarray, img2: np.ndarray):
        assert img1.shape == img2.shape
        diff = cv.absdiff(img1, img2)
        plt.imshow(diff, cmap='gray')
        plt.show()

    @staticmethod
    def _basic_processing(image):
        image = (image[::, ::, 1] * 2.3) - (image[::, ::, 0] * 1.7)
        image = image * 255 / image.max()
        image = image.clip(0, 255)
        image = image.astype('uint8')
        return image

    @staticmethod
    def _get_fragment_features(data: tuple):
        fragment, y, x = data
        features = [
            y,
            x,
            fragment.mean(),
            np.median(fragment),
            np.min(fragment),
            np.max(fragment)]

        moments = cv.moments(fragment)

        log = lambda i: -1 * np.copysign(1.0, i) * np.log10(abs(i)) if i != 0 else 0

        # features.extend(map(log, moments))
        features.extend(map(log, cv.HuMoments(moments)))
        return np.array(features)

    @staticmethod
    def _split_image(image: np.ndarray, pro_mask: np.ndarray = None, step: int = 5):
        for y in range(0, image.shape[0] - 4, step):
            for x in range(0, image.shape[1] - 4, step):
                if pro_mask is None:
                    yield image[y:y + 5, x:x + 5], y + 2, x + 2
                else:
                    yield image[y:y + 5, x:x + 5], y + 2, x + 2, pro_mask[y + 2, x + 2] > 127

    def _prepare_dataset(self):
        print("divided data not found")
        m = Manager()
        results_negative = m.list()
        results_positive = m.list()

        def _process_file(i, results_positive, results_negative):
            # r = []
            source = cv.imread(f"images/{i:02d}_h.jpg")
            source = RetinaSegmentator._basic_processing(source)
            mask = cv.imread(f"images/masked/{i:02d}_h.tif", cv.IMREAD_GRAYSCALE)
            for fragment, y, x, label in self._split_image(source, mask):
                if label:
                    results_positive.append((self._get_fragment_features((fragment, y, x)), label))
                else:
                    results_negative.append((self._get_fragment_features((fragment, y, x)), label))

            # result.extend(r)

        processes = []
        for i in range(1, 16):
            p = Process(target=_process_file, args=(i, results_positive, results_negative))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        # dump(result, 'splitted_images.joblib')
        # print("dumped!")
        results_negative = np.array(results_negative)
        results_positive.extend(
            results_negative[np.random.randint(results_negative.shape[0], size=len(results_positive)), :])

        dataset, labels = list(zip(*results_positive))

        return dataset, labels

    def train_model(self):
        # self._training_data = load("training_data.joblib")
        print("Preparing")
        dataset, labels = self._prepare_dataset()
        print("Splitting")
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=0.2)
        # X_train, Y_train = dataset, labels
        print("Scaling")
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # self._training_data = X_test, X_train, Y_test, Y_train, scaler
        # dump(self._training_data, "training_data.joblib")
        # X_test, X_train, Y_test, Y_train, scaler = self._training_data

        print("Fitting")
        best = 0.87
        for i in range(100):
            clf = MLPClassifier(hidden_layer_sizes=(15, 7, 2), max_iter=500)
            clf.fit(X_train, Y_train)
            # scores = cross_val_score(clf, X_train, Y_train, cv=6, n_jobs=-1)
            # print(scores)
            score = clf.score(X_test, Y_test)
            print(score)
            if score > best:
                best = score
            dump(scaler, f"scaler{score}.joblib")
            dump(clf, f"model{score}.joblib")

    def nn_reconstruction(self, path, model=None, scaler=None):
        if model is None:
            model = "model.joblib"
            scaler = "scaler.joblib"

        clf = load(model)
        scaler = load(scaler)
        imo = cv.imread(path)
        im = self._basic_processing(imo)
        imp = np.pad(im, 2, constant_values=0)
        split = self._split_image(imp, step=1)
        print("Calculating features")
        data = self._pool.map(self._get_fragment_features, split)
        data = scaler.transform(data)
        print("Prediction")
        points = clf.predict(data)
        points.shape = np.array(im.shape)

        imo[points] = [0, 255, 255]
        plt.imshow(imo)
        plt.show()

    def __del__(self):
        self._pool.close()


if __name__ == '__main__':
    r = RetinaSegmentator()
    r.train_model()
    # r.nn_reconstruction("images/01_h.jpg", "modelexy/model0.8275350497347662.joblib", "modelexy/scaler0.8275350497347662.joblib")
    # for i in range(1, 16):
    #     r.nn_reconstruction(f"images/{i:02d}_h.jpg")
