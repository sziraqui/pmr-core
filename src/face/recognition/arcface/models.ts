import * as tf from '@tensorflow/tfjs-node';

export class ArcFaceResnetModel {
    constructor(
        private type: 'ir' | 'se_ir',
        private batchShape: [],
        private depth: number,
        private weightInit: 'string',
        private trainable: boolean) {

    }
    createResnet() {
        const inputLayer = tf.layers.input({ batchShape: this.batchShape, dtype: 'float32' });
        const conv1 = tf.layers.conv2d({
            filters: 64,
            kernelSize: [3, 3],
            strides: [1, 1],
            kernelInitializer: this.weightInit,
            name: 'conv1'
        }).apply(inputLayer);
        const bnorm1 = tf.layers.batchNormalization(this.getBnormConfig()).apply(conv1);
        const prelu1 = tf.layers.prelu({ name: 'prelu1' }).apply(bnorm1);
        const topModel = tf.model({ inputs: inputLayer, outputs: prelu1 as tf.SymbolicTensor });
        // ? pooling either here or in resnetblock conditionally

        // blocks
        const bnorm2 = tf.layers.batchNormalization(this.getBnormConfig()); // .apply(lastBlock)
        // ? get ouput of bnorm2
    }
    resnetV1Layers(upperModel: tf.layers.Layer, depth: number, bottleneckDepth: number, stride: number, rate: number = 1) {
        // ? get output shape of topLayer
        let shortcut;

        if (upperModel.outputShape.slice(-1)[0] == depth) {
            // max pool
            shortcut = this.subsample(upperModel.output as tf.SymbolicTensor, stride);
        } else {
            // conv2d and bnorm
            shortcut = tf.layers.conv2d({
                filters: depth,
                kernelSize: [1, 1],
                strides: [stride, stride],
                activation: null,
                kernelInitializer: this.weightInit
            }).apply(upperModel.output);

            shortcut = tf.layers.batchNormalization(this.getBnormConfig()).apply(shortcut);
        }
        // bottleneck layer 1
        let residual = tf.layers.batchNormalization(this.getBnormConfig()).apply(upperModel.output);

        residual = tf.layers.conv2d({
            filters: bottleneckDepth,
            kernelSize: [3, 3],
            strides: [1, 1],
            activation: null,
            kernelInitializer: this.weightInit
        }).apply(residual);

        residual = tf.layers.batchNormalization({
            momentum: 0.9,
            epsilon: 2e-5,
            betaInitializer: 'zeros',
            gammaInitializer: tf.initializers.randomNormal({
                mean: 1.0,
                stddev: 0.002
            }),
            trainable: this.trainable,
        }).apply(residual);

        // bottleneck prelu
        residual = tf.layers.prelu({}).apply(residual);
        // bottleneck layer 2

    }

    subsample(inputs: tf.SymbolicTensor, factor, name: string = null) {
        if (factor > 1) {
            return tf.layers.maxPooling2d({
                poolSize: [1, 1],
                strides: factor
            }).apply(inputs);
        }
        return inputs;
    }

    conv2dsame(upperLayer: tf.layers.Layer, numOuputs, kernelSize: number, strides: number, dilationRate: number) {
        let net;
        if (strides == 1) {
            if (dilationRate == 1) {
                net = tf.layers.conv2d({
                    filters: numOuputs,
                    kernelSize: [kernelSize, kernelSize],
                    strides: [strides, strides],
                    biasInitializer: null,
                    kernelInitializer: this.weightInit,
                    padding: 'same'
                }).apply(upperLayer.output);

                net = tf.layers.batchNormalization(this.getBnormConfig()).apply(net);
            } else {
                net = tf.layers.conv2d({
                    filters: numOuputs,
                    kernelSize: [kernelSize, kernelSize],
                    strides: [strides, strides],
                    biasInitializer: null,
                    dilationRate: dilationRate,
                    kernelInitializer: this.weightInit,
                    padding: 'same'
                }).apply(upperLayer.output);
            }
        }
    }

    private getBnormConfig() {
        return {
            axis: 3,
            momentum: 0.9,
            epsilon: 2e-5,
            betaInitializer: 'zeros',
            gammaInitializer: tf.initializers.randomNormal({
                mean: 1.0,
                stddev: 0.002
            }),
            trainable: this.trainable,
        };
    }
}
