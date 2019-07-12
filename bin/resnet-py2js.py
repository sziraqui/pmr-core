#!/usr/bin/env python3


import keras_applications
import keras
import argparse
import tensorflowjs as tfjs
from keras_applications import resnet, resnet_v2
import sys
import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--ver', '-v', help='ResNet version 1 or 2',
                    default=1, type=int, required=False)
parser.add_argument('--layers', '-l', help='Layers 50, 101 or 152',
                    type=int, default=50, required=False)
parser.add_argument('--inputShape', '-s', help='Input image shape',
                    default=[224, 224, 3], nargs=3, type=int, required=False)
parser.add_argument('--outputDir', '-o', help='Output dir to save js model with weights',
                    default='modelDir', required=False)

args = parser.parse_args()

model = None
# # Workaround to use models from keras_application
modelKwargs = {
    'backend': keras.backend,
    'layers': keras.layers,
    'models': keras.models,
    'utils': keras.utils
}

if args.ver == 1:
    if args.layers == 50:
        model = resnet.ResNet50(
            include_top=False, input_shape=tuple(args.inputShape), **modelKwargs)
    elif args.layers == 101:
        model = resnet.ResNet101(
            include_top=False, weights='imagenet')
    elif args.layers == 152:
        model = resnet.ResNet152(
            include_top=False, weights='imagenet', input_shape=tuple(args.inputShape), **modelKwargs)
    else:
        print('Unsupported ResNet config')

elif args.ver == 2:
    if args.layers == 50:
        model = resnet_v2.ResNet50V2(
            include_top=False, weights='imagenet', input_shape=tuple(args.inputShape), **modelKwargs)
    elif args.layers == 101:
        model = resnet_v2.ResNet101V2(
            include_top=False, weights='imagenet', input_shape=tuple(args.inputShape), **modelKwargs)
    elif args.layers == 152:
        model = resnet_v2.ResNet152V2(
            include_top=False, weights='imagenet', input_shape=tuple(args.inputShape), **modelKwargs)
    else:
        print('Unsupported ResNet config')
        sys.exit(1)
else:
    print('Unsupported ResNet config')
    sys.exit(1)

tfjs.converters.save_keras_model(model, args.outputDir)
