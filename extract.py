import mxnet as mx
import numpy as np
import pickle
import cv2
import os

cifarFolder = "cifar-10-batches-py"

def extractImagesAndLabels(path, file):
    fullpath = os.path.join(path, file)
    f = open(fullpath, 'rb')
    dict = pickle.load(f, encoding='bytes')
    images = dict[b'data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict[b'labels']
    imagearray = mx.nd.array(images)
    labelarray = mx.nd.array(labels)
    return imagearray, labelarray

def extractCategories(path, file):
    fullpath = os.path.join(path, file)
    f = open(fullpath, 'rb')
    dict = pickle.load(f, encoding='bytes')
    return dict[b'label_names']

def saveCifarImage(array, path, file):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.asnumpy().transpose(1,2,0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path+file+".png", array)

categories = extractCategories(cifarFolder, "batches.meta")

for j in range(1, 5):
    imgarray, lblarray = extractImagesAndLabels(cifarFolder, "data_batch_" + str(j))

    for i in range(0, len(imgarray)):
        category = lblarray[i].asnumpy()
        category = (int)(category[0])

        imageNumber = (i+1)*j

        FOLDER = "./images/" + str(categories[category].decode("utf-8"))
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER, exist_ok=True)

        saveCifarImage(imgarray[i], FOLDER + "/", "image"+(str)(imageNumber))