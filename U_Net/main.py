import os
import time
import tensorflow as tf
from network_bn import *


def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_integer('max_epoch', 20, '# of step in an epoch')
    flags.DEFINE_integer('test_step', 10, '# of step to test a model')
    flags.DEFINE_integer('save_step', 20000, '# of step to save a model')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    # data
    flags.DEFINE_integer('batch', 5, 'batch size')
    flags.DEFINE_integer('test_batch', 1, 'batch size')
    flags.DEFINE_integer('channel', 1, 'channel size')
    flags.DEFINE_integer('height', 256, 'height size')
    flags.DEFINE_integer('width', 256, 'width size')
    # Debug
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_string('log_dir','log','log directory')
    flags.DEFINE_boolean('is_train', True, 'Training or testing')
    flags.DEFINE_string('log_level', 'INFO', 'Log level')
    flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
    flags.DEFINE_integer('reload_step', 0, 'Reload step')
    # network
    flags.DEFINE_integer('network_depth', 5, 'network depth for U-Net')
    flags.DEFINE_integer('class_num', 2, 'output class number')
    flags.DEFINE_integer('start_channel_num', 64, 'start number of outputs')
    flags.DEFINE_boolean('use_gpu', False, 'use GPU or not')
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def main(_):
    conf = configure()
    sess = tf.Session()
    model = GAN(sess, conf)
    model.train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    tf.app.run()