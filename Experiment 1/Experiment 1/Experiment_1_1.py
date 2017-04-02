
# coding: utf-8

# In[1]:


# CrackNet.py : It is a deep learning network created by @leopauly (cnlp@leeds.ac.uk)
# used for identifying cracked patches in road images
# The networks solely build to imporve the precision & accuracy than the previous system

#get_ipython().magic('matplotlib inline')
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

from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#from keras.callbacks import History, CSVLogger
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras import backend as K

# In[2]:

seed = 7
numpy.random.seed(seed)


# In[2]:

# In[3]:
# preperation of test dataset
test_imagefolderpath = ('./random_test_new/')
test_imagePath = glob.glob(test_imagefolderpath + '/*.jpg')
X_test = numpy.array([numpy.array(Image.open(str(test_imagefolderpath + str(i) + '.jpg')).convert('RGB'), 'f') for i in range(0, 200000)])

print('loading finished')

img_rows = 99;
img_cols = 99;
channel = 3;
nb_classes = 2

K.set_image_dim_ordering('tf')
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channel)

# print (X_test.shape)

y_1_test = numpy.array([(0) for i in range(0, 100000)]);
y_2_test = numpy.array([(1) for i in range(100000, 200000)]);
y_test = numpy.append(y_1_test, y_2_test)

# print(y_1_test.shape)
# print(y_2_test.shape)
# print(y_test.shape)

# check_val2=2500
# myimage = array_to_img(X_test[check_val2])
# pyplot.imshow(myimage)
# pyplot.show()
# print (y_test[check_val2])

X_test = X_test.astype('float32')
X_test /= 255.
y_test = np_utils.to_categorical(y_test, nb_classes)

# print (y_test.shape)

# def checker_test(check):
#    myimage = array_to_img(X_test[check])
#    pyplot.imshow(myimage)
#    pyplot.show()
#    print(y_test[check])

# checker_test(2499)


# In[ ]:

# load json file and load the model
json_file = open('Network_1_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("Network_1_1.h5")
print("Loaded model from disk")

LeedsNet = loaded_model
LeedsNet.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


# In[ ]:

# Evaluated on test dataset : Checking if the model is loaded properly
#scores = LeedsNet.evaluate(X_test, y_test, verbose=1)
#print("Accuracy: %.2f%%" % (scores[1] * 100))

# In[9]:

# Evaluated on images taken from same dataset but not used for training or validation
scores = LeedsNet.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

# In[10]:

# Evaluated on images taken from same dataset but not used for training or validation
scores_c = LeedsNet.evaluate(X_test[100000:200000], y_test[100000:200000], verbose=0)
print("Accuracy of crack: %.2f%%" % (scores_c[1] * 100))
print("Recall: %.2f%%" % (scores_c[1] * 100))

# In[11]:

# Evaluated on images taken from same dataset but not used for training or validation
scores_b = LeedsNet.evaluate(X_test[0:100000], y_test[0:100000], verbose=0)
temp = (1 - scores_b[1])
Precision = scores_c[1] / (temp + scores_c[1])
print("Accuracy of bg: %.2f%%" % (scores_b[1] * 100))
print(" Precision: %.2f%%" % (Precision * 100))


# In[ ]:



