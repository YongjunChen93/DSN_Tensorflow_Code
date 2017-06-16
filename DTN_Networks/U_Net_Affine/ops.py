import tensorflow as tf
import numpy as np

def conv2d(inputs,num_outputs,kernel_size,scope,data_format):
    outputs = tf.contrib.layers.conv2d(
        inputs, num_outputs, kernel_size, scope=scope,
        data_format=data_format, activation_fn=None, biases_initializer=None)
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, center=True, activation_fn=tf.nn.relu,
        updates_collections=None, epsilon=1e-5, scope=scope+'/batch_norm',
        data_format=data_format)

def pool2d(inputs, kernel_size,scope,data_format,stride=[2,2]):
    return tf.contrib.layers.max_pool2d(
        inputs,kernel_size,scope=scope,padding='SAME',
        data_format=data_format,stride=stride)

def fully_connected(inputs, num_outputs, batch_size, scope):
    inputs = tf.reshape(inputs,[batch_size,-1],name=scope+'/reshape')
    return tf.contrib.layers.fully_connected(inputs,num_outputs,scope=scope+'fully_connect',activation_fn=None)

def batch_norm(inputs,scope,data_format,momentum=0.9,epsilon=1e-5,scale=True):
    return tf.contrib.layers.batch_norm(inputs,decay=momentum,updates_collections=None,epsilon=epsilon,scale=scale,scope=scope,data_format=data_format)

def deconv2d(inputs,num_outputs,kernel_size,scope,data_format,stride=[2,2],padding='SAME',activation_fn=tf.nn.relu):
    outputs =  tf.contrib.layers.conv2d_transpose(
        inputs,num_outputs,kernel_size,stride=stride,padding=padding,scope=scope,
        data_format = data_format,activation_fn=None, biases_initializer=None)
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, center=True, activation_fn=tf.nn.relu,
        updates_collections=None, epsilon=1e-5, scope=scope+'/batch_norm',
        data_format=data_format)

'''
class batch_norm(object):
    def __init__(self, epsilon = 1e-5, momentum = 0.9, name='batch_norm'):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,decay=self.momentum, updates_collections = None, epsilon=self.epsilon, scale = True, scope = self.name)
'''


