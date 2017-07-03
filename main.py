#! /usr/bin/env python

import os
import re
from time import sleep
import numpy as np
import scipy.ndimage
import keras
import keras.applications.vgg16
from keras.optimizers import SGD
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

def vggTrain(x_train, y_train):

    model = keras.applications.vgg16.VGG16(include_top=False, weights=None, input_tensor=None, input_shape=(128, 128, 3))

    x = Dense(8, activation='softmax', name='predictions')#(model.layers[-2].output)

    print model.summary()

    #Then create the corresponding model 
    my_model = Model(input=model.input, output=x)
    # print my_model.summary()

    # input = Input(shape=(3,200,200),name = 'image_input')

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    # model.fit(x_train, y_train, batch_size=32)


def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(128,128, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096 / 4, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096 / 8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    print model.summary()

    if weights_path:
        model.load_weights(weights_path)

    return model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        # rotation_range=40,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        rescale=1./255,
        # shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

def importData(dirList):
    pass

    keyVal = 0
    x_train = np.array([])
    y_train = np.array([])

    for key in dirList.keys():
        fileList = dirList[key]
        count = 0
        for file in fileList:
            print "reading " + file
            a = scipy.ndimage.imread(file)
            a = a.reshape(np.append(a.shape, 1) )
            a = np.tile(a, ( 1, 1, 3))
            if x_train.size == 0 and y_train.size == 0:
                x_train = a.reshape(np.append(1, a.shape))
                y_train = np.array([keyVal])
            else:
                assert x_train.size != 0
                assert y_train.size != 0
                x_train = np.append(x_train, a.reshape(np.append(1, a.shape)), 0 )
                y_train = np.append(y_train, keyVal)

            print x_train.shape
            print y_train.shape

            count += 1
            if count == 10:
                break


        keyVal += 1

    retList = [x_train, y_train]
    return retList


if __name__ == "__main__":
    print "main"
    VGG_16()

    rootDir = '/home/lewisbp/DATA/cycleGanSAR/copied_data'

    dirs = os.listdir(rootDir)

    regex = re.compile(r'\.git/?')
    filtered = filter(lambda i: not regex.search(i), dirs)

    dataList = {}

    for eachDir in filtered:
        fileList = []
        # print eachDir
        for root, dirs, files in os.walk(os.path.join(rootDir, eachDir) ):
            # print root
            # print os.path.join(root, files)
            if re.search(r'real', root) and re.search(r'train', root):
                print root
                for eachFile in files:
                    fileList.append(os.path.join(root, eachFile))
        dataList[eachDir] = fileList

    # import pprint
    # pp = pprint.PrettyPrinter(indent=4)
    for key in dataList.keys():
        # pp.pprint(dataList[key])
        print len(dataList[key])
    # sleep(5)

    # vals = importData(dataList)

    # vggTrain(vals[0], vals[1])

    # for root, dirs, files in os.walk(rootDir):
    #     if not re.search(r'/\.git/?', root):
    #         print root
    #         print dirs
        # print dirs
        # print files