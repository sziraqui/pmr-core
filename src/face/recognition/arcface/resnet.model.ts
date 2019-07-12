import * as tf from '@tensorflow/tfjs-node';
import { Utils } from '../../../utils'

export namespace ResNet {

    interface Config {
        version: 1 | 2,
        layers: 50 | 101 | 152,
        classes: number,
        inputShape: [number, number, 3]
    }

    function makeModelName(resnetConfig: Config) {
        return `resnet_v${resnetConfig.version}_${resnetConfig.layers}_${resnetConfig.inputShape[0]}x${resnetConfig.inputShape[1]}`;
    }

    async function loadResNet(resnetConfig: Config) {
        const modelName = makeModelName(resnetConfig);
        return await Utils.loadKerasModel(modelName);
    }
}