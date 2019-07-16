import * as tf from '@tensorflow/tfjs-node';
import * as path from 'path';
import { Image } from 'nodoface';

export namespace Utils {
    export const ROOT_DIR = path.resolve(path.join(__dirname, '..'));
    export const MODEL_DIR = path.join(ROOT_DIR, 'models');
    const FILE_PREFIX = 'file://';

    export async function loadKerasModel(modelName: string) {
        return await tf.loadLayersModel(path.join(FILE_PREFIX + MODEL_DIR, modelName, 'model.json'));
    }

    export function imageToTensor(image: Image) {
        const flatImg = image.toUint8Array();
        const shape = [image.height(), image.width(), image.channels()];
        let arr = Float32Array.from(flatImg);
        return tf.tensor(arr, shape, 'float32');
    }
}
