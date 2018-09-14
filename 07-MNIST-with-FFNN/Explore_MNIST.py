import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataPath = "../mnist-data/"

if not os.path.exists(dataPath):
    os.makedirs(dataPath)

input = input_data.read_data_sets(dataPath, one_hot=True)

print(input.train.images.shape)
print(input.train.labels.shape)
print(input.test.images.shape)
print(input.test.labels.shape)

image_0 = input.train.images[0]
image_0 = np.resize(image_0, (28, 28))
label_0 = input.train.labels[0]
print(label_0)

plt.imshow(image_0, cmap='Greys_r')
plt.show()
