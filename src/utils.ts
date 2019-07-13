import * as tf from '@tensorflow/tfjs-node';
import * as path from 'path';

export namespace Utils {
    export const ROOT_DIR = path.resolve(path.join(__dirname, '..'));
    export const MODEL_DIR = path.join(ROOT_DIR, 'models');
    const FILE_PREFIX = 'file:://';

    export async function loadKerasModel(modelName: string) {
        return await tf.loadLayersModel(path.join(FILE_PREFIX + MODEL_DIR, modelName, 'model.json'));
    }

    export async function loadFrozenModel(modelName: string) {
        const modelDef = FILE_PREFIX + path.join(MODEL_DIR, modelName, 'tensorflowjs_model.pb');
        const weights = FILE_PREFIX + path.join(MODEL_DIR, modelName, 'weights_manifest.json');
        return await tf.loadFrozenModel(modelDef, weights);
    }

}
