import * as tf from '@tensorflow/tfjs-node';
import * as path from 'path';

export namespace Utils {
    export const ROOT_DIR = path.resolve(path.join(__dirname, '..'));
    export const MODEL_DIR = path.join(ROOT_DIR, 'models');

    export async function loadKerasModel(modelName: string) {
        return await tf.loadLayersModel(path.join(MODEL_DIR, modelName, 'model.json'));
    }
}
