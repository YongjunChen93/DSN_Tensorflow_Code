import os
import time
import datetime
import tensorflow as tf
from network_bn import *

def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_integer('max_epoch', 20000, '# of step in an epoch')
    flags.DEFINE_integer('test_step', 1, '# of step to test a model')
    flags.DEFINE_integer('save_step', 1000, '# of step to save a model')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    # data
    flags.DEFINE_integer('batch', 2, 'batch size')
    flags.DEFINE_integer('test_batch', 10, 'batch size')
    flags.DEFINE_integer('channel', 1, 'channel size')
    flags.DEFINE_integer('height', 224, 'height size')
    flags.DEFINE_integer('width', 224, 'width size')
    # Debug
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_string('log_dir','log','log directory')
    flags.DEFINE_boolean('is_train', True, 'Training or testing')
    flags.DEFINE_string('log_level', 'INFO', 'Log level')
    flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
    flags.DEFINE_integer('reload_step', 0, 'Reload step')
    flags.DEFINE_integer('reload_stride', 1, 'Reload stride')
    flags.DEFINE_integer('reload_end', 0, 'Reload end')
    # network
    flags.DEFINE_integer('network_depth', 5, 'network depth for U-Net')
    flags.DEFINE_integer('class_num', 2, 'output class number')
    flags.DEFINE_integer('start_channel_num', 8, 'start number of outputs')
    flags.DEFINE_boolean('use_gpu', False, 'use GPU or not')
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS

def main(_):
    start = time.clock()
    conf = configure()
    sess = tf.Session()
    model = Unet(sess, conf)
    if conf.is_train == True:
        model.train()
        pass
    else:
        for i in range(conf.reload_step,conf.reload_end+1,conf.reload_stride):
            model.prediction()
            conf.reload_step = conf.reload_step + conf.reload_stride
        os.system("matlab -r 'run Afterprediction.m'")        
    end = time.clock()
    print((end-start)/60)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    tf.app.run()
