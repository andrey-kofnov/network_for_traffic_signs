import numpy as np

import cv2

import tensorflow as tf
from tensorflow.contrib.layers import flatten

import warnings

import os
from os import listdir
from os.path import isfile, join

from PIL import Image
import requests
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

import random


def preProc_t1(image, kernel_size = 1):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    return image

def preProc_t2(image):
    return ((image - 128.0)/128.0)

def preProc_t3(image):
    return image / 255.0


def train_test_union(train, test):
    t1 = list(test)
    t1.extend(list(train))
    t1 = [t1[i] for i in range(len(t1))]
    return np.array(list(set(t1))).reshape((np.array(list(set(t1))).shape[0],1))


def train_test_union_v2(train, test):
    t1 = list(test)
    t1.extend(list(train))
    t1 = [t1[i][0] for i in range(len(t1))]
    return np.array(list(set(t1))).reshape((np.array(list(set(t1))).shape[0],1))


def list_Diff(li1, li2): 
    return (list(set(li1) - set(li2)))


def summary_statistics(data, column, n_cl):
    dat = pd.DataFrame(data, columns=[column])
    grouping = dat.groupby([column])[column].count()
    dat.hist(bins=n_cl) 
    print(column + ' summary statistics:\n{}'.format(round(grouping.describe(percentiles=[]),0)))



    ### Model

def neuNet(x, kp, n_cl):    
    
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. 
    tmp1_We = tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 10), mean = mu, stddev = sigma))
    tmp1_fv = tf.Variable(tf.zeros(10))
    tmp1   = tf.nn.conv2d(x, tmp1_We, strides=[1, 1, 1, 1], padding='VALID') + tmp1_fv

    # Activation.
    tmp1 = tf.nn.relu(tmp1)

    # Pooling.
    tmp1 = tf.nn.max_pool(tmp1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Layer 2: Convolutional.
    tmp2_We = tf.Variable(tf.truncated_normal(shape=(4, 4, 10, 30), mean = mu, stddev = sigma))
    tmp2_fv = tf.Variable(tf.zeros(30))
    tmp2   = tf.nn.conv2d(tmp1, tmp2_We, strides=[1, 1, 1, 1], padding='VALID') + tmp2_fv
    
    # Activation.
    tmp2 = tf.nn.relu(tmp2)

    # Pooling. 
    tmp2 = tf.nn.max_pool(tmp2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    tmp2 = tf.nn.dropout(tmp2, kp)


    fc0   = flatten(tmp2)
    
    # Layer 3: Fully Connected. 
    tmp1_We = tf.Variable(tf.truncated_normal(shape=(1080, 270), mean = mu, stddev = sigma))
    tmp1_fv = tf.Variable(tf.zeros(270))
    tmp1   = tf.matmul(fc0, tmp1_We) + tmp1_fv
    
    # Activation.
    tmp1    = tf.nn.relu(tmp1)

    # Layer 4: Fully Connected. 
    tmp2_We  = tf.Variable(tf.truncated_normal(shape=(270, 129), mean = mu, stddev = sigma))
    tmp2_fv  = tf.Variable(tf.zeros(129))
    tmp2    = tf.matmul(tmp1, tmp2_We) + tmp2_fv
    
    # Activation.
    tmp2    = tf.nn.relu(tmp2)

    # Layer 5: Fully Connected. 
    tmp3_We  = tf.Variable(tf.truncated_normal(shape=(129, n_cl), mean = mu, stddev = sigma))
    tmp3_fv  = tf.Variable(tf.zeros(n_cl))
    logits = tf.matmul(tmp2, tmp3_We) + tmp3_fv
    
    return logits




def evaluate(X_data, y_data, batch, ac_oper, x, y, kp):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch):
        batch_x, batch_y = X_data[offset:offset+batch], y_data[offset:offset+batch]
        accuracy = sess.run(ac_oper, feed_dict={x: batch_x, y: batch_y, kp: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



def download_image(url):
    img = Image.open(requests.get(url, stream=True).raw)
    return np.array(img)