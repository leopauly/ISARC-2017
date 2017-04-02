
# CrackNet.py : It is a deep learning network created by @leopauly (cnlp@leeds.ac.uk)
# used for identifying cracked patches in road images
# The networks for evaluating deep layers vs accuracy in 200,000 patches

import matplotlib
matplotlib.use('Agg')
import numpy
import glob
from PIL import Image
import datetime
import sys
from scipy import misc
from scipy import ndimage
from matplotlib import pyplot
from sklearn.cross_validation import train_test_split
from keras.models import model_from_json
from sklearn.utils import shuffle
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dense

from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import History, CSVLogger
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# In[ ]:

seed = 7
numpy.random.seed(seed)


# In[ ]: Load data base and give image propereties like hight, width etc.

imagefolderpath = ('./random_train_new/')
imagePath = glob.glob(imagefolderpath + '/*.jpg')
X = numpy.array(
    [numpy.array(Image.open(str(imagefolderpath + str(i) + '.jpg')).convert('RGB'), 'f') for i in range(0, 40000)])

img_rows = 99;
img_cols = 99;
channel = 3;
nb_classes = 2

# In[ ]:

y_1 = numpy.array([(0) for i in range(0, 20000)]);
y_2 = numpy.array([(1) for i in range(20000, 40000)]);
y = numpy.append(y_1, y_2)


X, y = shuffle(X, y, random_state=1)
X, y = shuffle(X, y, random_state=2)
X, y = shuffle(X, y, random_state=3)
X, y = shuffle(X, y, random_state=4)
X, y = shuffle(X, y, random_state=5)
X, y = shuffle(X, y, random_state=6)
X, y = shuffle(X, y, random_state=7)
X, y = shuffle(X, y, random_state=8)
X, y = shuffle(X, y, random_state=9)
X, y = shuffle(X, y, random_state=10)
X, y = shuffle(X, y, random_state=11)
X, y = shuffle(X, y, random_state=12)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.0, random_state=4)

# In[ ]:

K.set_image_dim_ordering('tf')
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channel)
input_shape = (img_rows, img_cols, channel)


X_train = X_train.astype('float32')
X_train /= 255.


y_train = np_utils.to_categorical(y_train, nb_classes)


# In[ ]: preperation of test dataset
#test_imagefolderpath = ('./test_raul/')
#test_imagePath = glob.glob(test_imagefolderpath + '/*.jpg')
#X_test = numpy.array([numpy.array(Image.open(str(test_imagefolderpath + str(i) + '.jpg')).convert('RGB'), 'f') for i in
#                      range(0, 200000)])

#img_rows = 99;
#img_cols = 99;
#channel = 3;
#nb_classes = 2

#K.set_image_dim_ordering('tf')
#X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channel)

# print (X_test.shape)

#y_1_test = numpy.array([(0) for i in range(0, 100000)]);
#y_2_test = numpy.array([(1) for i in range(100000, 200000)]);
#y_test = numpy.append(y_1_test, y_2_test)

# print(y_1_test.shape)
# print(y_2_test.shape)
# print(y_test.shape)

# check_val2=2500
# myimage = array_to_img(X_test[check_val2])
# pyplot.imshow(myimage)
# pyplot.show()
# print (y_test[check_val2])

#X_test = X_test.astype('float32')
#X_test /= 255.
#y_test = np_utils.to_categorical(y_test, nb_classes)

# print (y_test.shape)

# def checker_test(check):
#    myimage = array_to_img(X_test[check])
#    pyplot.imshow(myimage)
#    pyplot.show()
#    print(y_test[check])

# checker_test(2499)


# In[ ]:




# In[ ]:

# load json file and load the model
json_file = open('Network_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("Network_1.h5")
print("Loaded model from disk")

LeedsNet = loaded_model
print (LeedsNet.summary())

# In[ ]:




# In[ ]:

# datagen = ImageDataGenerator(
#    featurewise_center=False,
#    featurewise_std_normalization=False,
#    rotation_range=20,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    horizontal_flip=True)
# datagen.fit(X_train)


# In[ ]:

epochs = 40
lrate =0.00009803921
sgd = SGD(lr=lrate, momentum=0.9,decay= .0005)
LeedsNet.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# In[ ]:

b_size = 48
# csv_logger = CSVLogger('trainingLeedsnet.log',separator=',', append=True)

start_time = datetime.datetime.now()
print (start_time)

LeedsNet.fit(X_train, y_train, nb_epoch=epochs,
             batch_size=b_size,
             shuffle='batch')  # validation_data=(X_new, y_new),validation_data=(X_test, y_test),

end_time = datetime.datetime.now()
print (end_time)


# storing model to a json file and weights in HDF5 format
model_json = LeedsNet.to_json()
# saving model
with open("Network_1_1.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5 and saving it
LeedsNet.save_weights("Network_1_1.h5")
print("Saved model to disk")