#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import tensorflow as tf
import keras as K
import h5py
import math
import time

from data import load_data
from model import unet

from argparser import args

intra=0
inter=0
if 'intra_op_parallelism_threads' in os.environ:
    intra = int(os.environ['intra_op_parallelism_threads'])
if 'inter_op_parallelism_threads' in os.environ:
    inter = int(os.environ['inter_op_parallelism_threads'])
if intra != 0 and inter != 0:
    config = tf.ConfigProto(intra_op_parallelism_threads=intra,
            inter_op_parallelism_threads=inter)
    session = tf.Session(config=config)
    K.backend.set_session(session)

print ("We are using Tensorflow version", tf.__version__, "with Intel(R) oneDNN", "enabled" if tf.pywrap_tensorflow.IsMklEnabled() else "disabled",)

data_path = os.path.join("../../data/decathlon/144x144/")
data_filename = "Task01_BrainTumour.h5"
hdf5_filename = os.path.join(data_path, data_filename)
imgs_train, msks_train, imgs_validation, msks_validation, imgs_testing, msks_testing = load_data(hdf5_filename)
imgs_warmup=imgs_testing[:10]
imgs_infere=imgs_testing[10:266]
print("Number of imgs_warmup: {}".format(imgs_warmup.shape[0]))
print("Number of imgs_infere: {}".format(imgs_infere.shape[0]))

unet_model = unet()
model = unet_model.load_model(os.path.join("./output/unet_model_for_decathlon.hdf5"))

def do_benchmark(batch_size=32):
    if 'OMP_NUM_THREADS' in os.environ:
        print('OMP_NUM_THREAD: {}'.format(os.environ['OMP_NUM_THREADS']))
    if 'KMP_BLOCKTIME' in os.environ:
        print('KMP_BLOCKTIME: {}'.format(os.environ['KMP_BLOCKTIME']))
    if 'KMP_AFFINITY' in os.environ:
        print('KMP_AFFINITY: {}'.format(os.environ['KMP_AFFINITY']))
    if 'KMP_SETTINGS' in os.environ:
        print('KMP_SETTINGS: {}'.format(os.environ['KMP_SETTINGS']))
    if 'intra_op_parallelism_threads' in os.environ:
        print('intra_op_parallelism_threads: {}'.format(os.environ['intra_op_parallelism_threads']))
    if 'inter_op_parallelism_threads' in os.environ:
        print('inter_op_parallelism_threads: {}'.format(os.environ['inter_op_parallelism_threads']))
    model.predict(imgs_warmup, batch_size, verbose=1, steps=None)
    t0 = time.time()
    model.predict(imgs_infere, batch_size, verbose=1, steps=None)
    t1 = time.time()
    num = math.ceil(imgs_infere.shape[0]/batch_size)
    lat=(t1-t0)/num
    thr=batch_size/lat
    print('Latency: {:.2f}ms; Throughput: {:.2f}fps'.format(lat*1000, thr))

print('batch size: {}'.format(args.batch_size))
do_benchmark(args.batch_size)
