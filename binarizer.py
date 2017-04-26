import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import lanelines.features as features
from lanelines.plotting import imagesc


"""
This script is used for training binarization classifier
"""

DISPLAY_TEST_IMAGES = False


def preprocess(img, cal, pt):
    undistorted = cv2.undistort(img, cal['matrix'], cal['distortion'])
    transformed = cv2.warpPerspective(undistorted, pt['matrix'], tuple(pt['target_size']), flags=cv2.INTER_LANCZOS4)
    return transformed


if __name__ == '__main__':
    # load calibration data
    cal_data = np.load('data/calibration.npz')
    # load perspective transform data
    pt_data = np.load('data/perspective.npz')

    images = [('train_images/test1.jpg', 'train_images/test1_mask.png'),
              ('train_images/test5.jpg', 'train_images/test5_mask.png'),
              ('train_images/straight_lines1.jpg', 'train_images/straight_lines1_mask.png'),
              ('train_images/harder_challenge_0005.png', 'train_images/harder_challenge_0005_mask.png'),
              ('train_images/harder_challenge_0009.png', 'train_images/harder_challenge_0009_mask.png'),
              ('train_images/harder_challenge_0010.png', 'train_images/harder_challenge_0010_mask.png'),
              ('train_images/harder_challenge_0011.png', 'train_images/harder_challenge_0011_mask.png'),
              ('train_images/harder_challenge_0013.png', 'train_images/harder_challenge_0013_mask.png'),
              ('train_images/harder_challenge_0014.png', 'train_images/harder_challenge_0014_mask.png'),
              ('train_images/harder_challenge_0001.png', 'train_images/harder_challenge_0001_mask.png'),
              ('train_images/harder_challenge_0002.png', 'train_images/harder_challenge_0002_mask.png'),
              ('train_images/harder_challenge_0003.png', 'train_images/harder_challenge_0003_mask.png')]

    x_data = np.empty((0, features.n_features), np.float)
    y_data = np.empty(0, np.int)

    # setup ml data
    print('Collecting data...')
    for image_filename, mask_filename in images:
        # load image
        image = preprocess(cv2.imread(image_filename), cal_data, pt_data)
        # load mask
        mask = preprocess(cv2.cvtColor(cv2.imread(mask_filename), cv2.COLOR_BGR2GRAY), cal_data, pt_data)
        mask = cv2.normalize(mask.astype(np.float), None, 0.0, 1.0, cv2.NORM_MINMAX)

        # extract features
        f = features.extract(image)

        # prepare data for training
        x_data_sample = f.reshape(-1, f.shape[-1])
        y_data_sample = mask.flatten() > 0.75

        # combine with previous
        x_data = np.concatenate([x_data, x_data_sample])
        y_data = np.concatenate([y_data, y_data_sample])

        # mirror augmentation
        mask = cv2.flip(mask, 1)
        image = cv2.flip(image, 1)
        f = features.extract(image)

        # prepare data for training
        x_data_sample = f.reshape(-1, f.shape[-1])
        y_data_sample = mask.flatten() > 0.75

        # combine with previous
        x_data = np.concatenate([x_data, x_data_sample])
        y_data = np.concatenate([y_data, y_data_sample])

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, stratify=y_data)

    # fit classifier
    print('Training classifier...')
    dt = RandomForestClassifier(max_depth=24, class_weight='balanced')
    dt.fit(x_train, y_train)

    # get results
    y_predicted = dt.predict(x_test)
    ps = precision_score(y_test, y_predicted)
    rs = recall_score(y_test, y_predicted)
    print('Precision: %f, Recall: %f' % (ps, rs))

    # save binarizer
    joblib.dump(dt, 'data/binarizer.clf')
    print('Saved trained classifier.')

    if DISPLAY_TEST_IMAGES:
        clf = joblib.load('data/binarizer.clf')
        test_images = ['test_images/straight_lines2.jpg',
                       'test_images/test2.jpg',
                       'test_images/test3.jpg',
                       'test_images/test4.jpg',
                       'test_images/test6.jpg',
                       'test_images/harder_challenge_0004.png',
                       'test_images/harder_challenge_0006.png',
                       'test_images/harder_challenge_0007.png',
                       'test_images/harder_challenge_0008.png',
                       'test_images/harder_challenge_0012.png']

        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Binary', cv2.WINDOW_NORMAL)
        for test_filename in test_images:
            image = preprocess(cv2.imread(test_filename), cal_data, pt_data)
            f = features.extract(image)
            res = clf.predict(f.reshape(-1, f.shape[-1]))
            cv2.imshow('Original', image)
            imagesc('Binary', res.reshape(image.shape[:2]))
            cv2.waitKey(0)

        cv2.destroyAllWindows()



