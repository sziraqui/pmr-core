import { readFileSync, existsSync } from 'fs';
import * as tf from '@tensorflow/tfjs-core';
import { utils } from '..';
import * as path from 'path';
import * as _ from 'lodash';
import * as knn from '../../../knn-classifier/src/index';
export const K = 4;

export class FaceClassifier {
    private static instance: FaceClassifier;
    private classifier;
    private labels: {};
    private constructor() {
        this.classifier = knn.create();
    }
    static async getInstance() {
        if (!FaceClassifier.instance) {
            FaceClassifier.instance = new FaceClassifier();
            await FaceClassifier.instance.load('knn_classifier_10examples');
        }
        return FaceClassifier.instance;
    }
    async loadWeights(dir: string) {

        if (existsSync(dir)) {
            let data = readFileSync(dir);
            return await JSON.parse(data.toString());
        } else {
            return Promise.reject(new Error(`Cannot find model file: ${dir}/model.json`));
        }
    }

    async load(modelName: string) {
        return this.loadWeights(path.join(utils.MODEL_DIR, modelName, 'model.json'))
            .then(weights => weights["data"])
            .then(dataList => this.initModel(dataList))
            .then(success => {
                if (success) console.log('Face classifier initialised');
                else console.log('Cannot initialise face classifier');
            })
            .catch(err => console.log(err.message));
    }

    async initModel(dataList: { embedding: number[], personname: string }[]) {
        let labels = {};
        let clasNum = 0;
        for (let data of dataList) {
            if (labels[data.personname] === undefined) {
                labels[data.personname] = clasNum;
                clasNum++;
            }
        }
        this.labels = _.invert(labels);
        await dataList.forEach((data) => {
            let tsr = tf.tensor1d(data.embedding);
            this.classifier.addExample(tsr, data.personname);
        });
        return await true;
    }

    async predict(embedding: Float64Array) {
        let res = await this.classifier.predictClass(tf.tensor1d(Array.from(embedding)), K);
        return res;
    }

    getNumClasses() {
        return this.classifier.getNumClasses();
    }
}