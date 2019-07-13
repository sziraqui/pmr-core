#!/bin/sh
function convert_frozen() {
    # $1 : path to insightface.pb file
    # $2 : output directory
    tensorflowjs_converter \
    --input_format=tf_frozen_model \
    --output_node_names='resnet_v1_50/E_DenseLayer/W' \
    --output_format=tensorflowjs $1 $2
}

convert_frozen $1 $2
