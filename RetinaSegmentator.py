import time
from itertools import chain
from multiprocessing import Manager, Pool
from multiprocessing import Process
import random

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from joblib import dump, load


class RetinaSegmentator:
    def __init__(self):
        self._pool = Pool()

    @classmethod
    def classical_deconstruction(cls, image):
        image = plt.imread(image)
        orig_image = image
        image = cls._basic_processing(image)

        kernel = np.ones((6, 6), np.uint8)
        image = cv.erode(image, kernel)
        kernel = np.ones((3, 3), np.uint8)
        image = cv.dilate(image, kernel)

        image = cv.bilateralFilter(image, 5, 210, 30)
        image = cv.medianBlur(image, 9)

        image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 41, 4)  # 61, 4

        image = cv.bilateralFilter(image, 5, 70, 10)
        image = cv.medianBlur(image, 9)

        im = np.array(orig_image)
        im[image > 50] = [0, 255, 255]
        plt.imshow(im, cmap='gray')
        plt.axis("off")
        plt.show()
        return image

    def reconstruct_and_compare(self, i):
        source = plt.imread(f"training/{i:02d}_h.jpg")
        mask = plt.imread(f"training/masked/{i:02d}_h.tif", cv.IMREAD_GRAYSCALE)

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
        image = (image[::, ::, 1] * 2.3) - (image[::, ::, 2] * 1.7)
        image = image * 255 / image.max()
        image = image.clip(0, 255)
        image = image.astype('uint8')
        return image

    @staticmethod
    def _get_fragment_features(data: tuple):
        fragment, y, x = data
        features = list(fragment.reshape(25))
        features.extend([
            fragment.mean(),
            np.median(fragment),
            np.min(fragment),
            np.max(fragment)])

        moments = cv.moments(fragment)

        log = lambda i: -1 * np.copysign(1.0, i) * np.log10(abs(i)) if i != 0 else 0

        # features.extend(map(log, moments))
        features.extend(map(log, cv.HuMoments(moments)))
        return features

    @staticmethod
    def _split_image(image: np.ndarray, pro_mask: np.ndarray = None, step: int = 3):
        for y in range(0, image.shape[0] - 4, step):
            for x in range(0, image.shape[1] - 4, step):
                if pro_mask is None:
                    yield image[y:y + 5, x:x + 5], y + 2, x + 2
                else:
                    yield image[y:y + 5, x:x + 5], y + 2, x + 2, pro_mask[y + 2, x + 2] > 127

    def _prepare_dataset(self, s):
        m = Manager()
        results_negative = m.list()
        results_positive = m.list()

        def _process_file(i, results_positive, results_negative):
            # r = []
            source, _, _ = self._read_image(f"training/{i:02d}.jpg")
            source = RetinaSegmentator._basic_processing(source)
            mask, _, _ = self._read_image(f"training/masked/{i:02d}.tif")
            for fragment, y, x, label in self._split_image(source, mask, step=2):
                if label:
                    results_positive.append((self._get_fragment_features((fragment, y, x)), label))
                else:
                    results_negative.append((self._get_fragment_features((fragment, y, x)), label))

            # result.extend(r)

        processes = []
        for i in range(1, 21):
            p = Process(target=_process_file, args=(i, results_positive, results_negative))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
            print(f"{p.pid} Finished")

        print("All processes finished")
        # dump(result, 'splitted_images.joblib')
        # print("dumped!")
        # results_negative = np.array(results_negative)
        print("To tuple", end=" ")
        t = time.time()
        results_positive = tuple(results_positive)
        # dump(results_positive, "/mnt/results_positive")
        # print("Dumped positive")
        results_negative = tuple(results_negative)
        # dump(results_negative, "/mnt/results_negative")
        # print("Dumped negative")
        print(f"took {time.time() - t}s")

        print(f"Results negative: {len(results_negative)}")
        print(f"Results positive: {len(results_positive)}")
        print(f"Desired negative sample: {len(results_positive) * s}")
        print("Shuffling", end=" ")
        t = time.time()
        result = chain(results_positive, random.sample(results_negative, len(results_positive) * s))
        print(f"took {time.time() - t}s")

        print("Zipping", end=" ")
        t = time.time()
        dataset, labels = tuple(zip(*result))
        print(f"took {time.time() - t}s")

        return dataset, labels

    def train_model(self, s):
        # self._training_data = load("training_data.joblib")
        print("Preparing")
        dataset, labels = self._prepare_dataset(s)

        print("Splitting")
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=0.2)
        # X_train, Y_train = dataset, labels
        print("Scaling")
        scaler = StandardScaler()
        scaler.fit(dataset)
        # dataset = scaler.transform(dataset)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        clf = MLPClassifier(hidden_layer_sizes=(15, 7, 2), max_iter=1000)
        print("Fitting")
        clf.fit(X_train, Y_train)
        # scores = cross_val_score(clf, X_train, Y_train, cv=6, n_jobs=-1)
        # print(scores)
        score = clf.score(X_test, Y_test)
        print(score)

        Y_predicted = clf.predict(X_test)

        print(classification_report(Y_test, Y_predicted))
        cm = confusion_matrix(Y_test, Y_predicted)
        print(cm)
        (tn, fp), (fn, tp) = cm
        print(f"Czułość: {tp / (tp + fn)}")
        print(f"Swoistość: {tn / (tn + fp)}")

        plot_confusion_matrix(clf, X_test, Y_test)
        plt.show()

        dump(scaler, f"models36resized/{s}/scaler{score}.joblib")
        dump(clf, f"models36resized/{s}/model{score}.joblib")
        print(f"Dumped model to models36resized/{s}/model{score}.joblib")

        del X_train, X_test, Y_train, Y_test
        return clf, scaler

    def nn_reconstruction(self, path, model=None, scaler=None):
        if model is None:
            model = "models36resized/4/model0.9153591930626876.joblib"
            scaler = "models36resized/4/scaler0.9153591930626876.joblib"

        if type(model) == str and type(scaler) == str:
            model = load(model)
            scaler = load(scaler)

        imo, vshape, hshape = self._read_image(path)
        im = self._basic_processing(imo)
        imp = np.pad(im, 2, constant_values=0)
        split = self._split_image(imp, step=1)

        # _proccess = lambda f: clf.predict(scaler.transform([self._get_fragment_features(f)]))[0]

        print("Calculating features")
        data = self._pool.map(self._get_fragment_features, split)
        data = scaler.transform(data)
        print("Prediction")
        points = model.predict(data)

        # points = np.array(tuple(map(_proccess, split)))
        points.shape = np.array(im.shape)

        imo[points] = [0, 255, 255]

        points = points.astype('uint8')
        mask = cv.resize(points, (hshape, vshape))
        mask = mask * 255

        # suffix = random.randint(1, 100000000)
        # plt.imsave(f"models36resized/mask{suffix}.png", mask, cmap="gray")

        imo = cv.resize(imo, (hshape, vshape))
        # plt.imsave(f"models36resized/imo{suffix}.png", imo, cmap="gray")

        plt.imshow(imo)
        plt.axis("off")
        plt.show()

        return mask

    @staticmethod
    def _read_image(path):
        im = plt.imread(path)
        im = np.array(im).astype('uint8')
        org_shape = im.shape
        im = cv.resize(im, (700, 700))
        return im, org_shape[0], org_shape[1]

    def _find_best_balance(self):
        for i in range(1, 8):
            clf, scaler = self.train_model(i)
            # for j in range(1, 3):
            #     rmask = self.nn_reconstruction(f"validate/{j:02d}.jpg", clf, scaler)
            #     pmask = plt.imread(f"validate/{j:02d}.tif")
            #     print(f"\t\tBalance 1:{i}\tTry {j}\tabsdiff: {np.sum(cv.absdiff(rmask, pmask))}")

    def test(self):
        for i in range(7, 15):
            pmask = np.array(plt.imread(f"test/{i:02d}.png")).astype("uint8")
            cmask = self.classical_deconstruction(f"test/{i:02d}.jpg")
            nmask = self.nn_reconstruction(f"test/{i:02d}.jpg", "models36resized/4/model0.9153591930626876.joblib",
                                           "models36resized/4/scaler0.9153591930626876.joblib")

            pmask = pmask * 255 / pmask.max()
            cmask = cmask * 255 / pmask.max()
            nmask = nmask * 255 / pmask.max()
            diff = cv.absdiff(cmask.astype("uint8"), nmask.astype("uint8")).astype("uint8")
            print(f"absdiff: {np.sum(diff)}")
            plt.imshow(diff, cmap="gray")
            plt.axis("off")
            plt.show()
            diff = cv.absdiff(cmask.astype("uint8"), pmask.astype("uint8")).astype("uint8")
            print(f"absdiff: {np.sum(diff)}")
            plt.imshow(diff, cmap="gray")
            plt.axis("off")
            plt.show()
            diff = cv.absdiff(nmask.astype("uint8"), pmask.astype("uint8")).astype("uint8")
            print(f"absdiff: {np.sum(diff)}")
            plt.imshow(diff, cmap="gray")
            plt.axis("off")
            plt.show()
            pass

    def __del__(self):
        self._pool.close()

    @classmethod
    def classical_deconstruction_mask(cls, image):
        image = plt.imread(image)
        orig_image = image
        image = cls._basic_processing(image)

        kernel = np.ones((6, 6), np.uint8)
        image = cv.erode(image, kernel)
        kernel = np.ones((3, 3), np.uint8)
        image = cv.dilate(image, kernel)

        image = cv.bilateralFilter(image, 5, 210, 30)
        image = cv.medianBlur(image, 9)

        image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 41, 4)  # 61, 4

        image = cv.bilateralFilter(image, 5, 70, 10)
        image = cv.medianBlur(image, 9)

        mask = np.zeros(image.shape)
        mask[image > 50] = 1

        return mask

    def classical_stats(self):
        mask = self.classical_deconstruction_mask("sprawko/3.jpg").astype("bool").flatten()
        pro_mask = plt.imread("sprawko/3.tif")
        pro_mask = np.array(pro_mask).astype('bool').flatten()
        cm = confusion_matrix(pro_mask, mask)
        print(cm)
        (tn, fp), (fn, tp) = cm
        print(f"Czułość: {tp / (tp + fn)}")
        print(f"Swoistość: {tn / (tn + fp)}")


if __name__ == '__main__':
    r = RetinaSegmentator()
    # r.test()
    # r._find_best_balance()
    # for s in range(2, 9):
    # r.train_model(4)
    # r.nn_reconstruction("sprawko/1.jpg", "models36resized/4/model0.9123775006557142.joblib", "models36resized/4/scaler0.9123775006557142.joblib")
    r.classical_stats()
    # for s in range(1, 16):
    #     r.nn_reconstruction(f"training/{s:02d}_h.jpg")
