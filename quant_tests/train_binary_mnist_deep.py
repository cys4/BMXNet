"""
Train mnist, see more explanation at http://mxnet.io/tutorials/python/mnist.html
"""
"""
A port of 'mnist_deep.py' in TensorFlow example
(https://github.com/tensorflow/tensorflow/blob/v1.3.0/tensorflow/examples/tutorials/mnist/mnist_deep.py)
"""
import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, fit
from common.util import download_file
import mxnet as mx
import numpy as np
import gzip, struct

NUM_BITS = 1

def read_data(label, image):
    """
    download and read data into numpy
    """
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    with gzip.open(download_file(base_url+label, os.path.join('data',label))) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_file(base_url+image, os.path.join('data',image)), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)


def to4d(img):
    """
    reshape to 4D arrays
    """
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

def get_mnist_iter(args, kv):
    """
    create data iterator with NDArrayIter
    """
    (train_lbl, train_img) = read_data(
            'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data(
            't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')
    train = mx.io.NDArrayIter(
        to4d(train_img), train_lbl, args.batch_size, shuffle=True)
    val = mx.io.NDArrayIter(
        to4d(val_img), val_lbl, args.batch_size)
    return (train, val)

def get_binary_mnist_deep(num_classes=10, add_stn=False, **kwargs):
    data = mx.symbol.Variable('data')
    #if add_stn:
    #    data = mx.sym.SpatialTransformer(data=data, loc=get_loc(data), target_shape = (28,28),
    #                                     transform_type="affine", sampler_type="bilinear")
    # first conv
    #conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=32)
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=64)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    #conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=64)
    #relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    #pool2 = mx.symbol.Pooling(data=relu2, pool_type="max",
    #                          kernel=(2,2), stride=(2,2))
    bact2 = mx.symbol.QActivation(data=pool1, backward_only=True, act_bit=NUM_BITS)
    conv2 = mx.symbol.QConvolution(data=bact2, kernel=(5,5), num_filter=64, act_bit=NUM_BITS)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    #fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=1024)
    #relu3 = mx.symbol.Activation(data=fc1, act_type="relu")
    bact3 = mx.symbol.QActivation(data=flatten, backward_only=True, act_bit=NUM_BITS)
    fc1 = mx.symbol.QFullyConnected(data=bact3, num_hidden=1024, act_bit=NUM_BITS)
    relu3 = mx.symbol.Activation(data=fc1, act_type="relu")

    # dropout
    dropout1 = mx.symbol.Dropout(data=relu3, p = 0.5)

    # second fullc
    fc2 = mx.symbol.FullyConnected(data=dropout1, num_hidden=num_classes)
    #fc2 = mx.symbol.FullyConnected(data=relu3, num_hidden=num_classes)

    # loss
    softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return softmax
if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-classes', type=int, default=10,
                        help='the number of classes')
    parser.add_argument('--num-examples', type=int, default=60000,
                        help='the number of training examples')
    
    #parser.add_argument('--add_stn',  action="store_true", default=False, help='Add Spatial Transformer Network Layer (lenet only)')
    
    fit.add_fit_args(parser)
    parser.set_defaults(
        # network
        network        = 'mlp',
        # train
        gpus           = None,
        batch_size     = 64,
        disp_batches   = 100,
        num_epochs     = 20,
        lr             = .05,
        lr_step_epochs = '10'
    )
    args = parser.parse_args()

    # load network
    #from importlib import import_module
    #net = import_module('symbols.'+args.network)
    #sym = net.get_symbol(**vars(args))
    sym = get_binary_mnist_deep(**vars(args))

    # train
    fit.fit(args, sym, get_mnist_iter)
