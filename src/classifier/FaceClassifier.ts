import { create as CreateClassifier, KNNClassifier } from '@tensorflow-models/knn-classifier';
import { readFileSync, existsSync } from 'fs';
import { utils } from '..';
import * as path from 'path';
import * as tf from '@tensorflow/tfjs-node';
import * as _ from 'lodash';

export const MAX_CLASSIFICATIONS = 10;

export class FaceClassifier {
    private static instance: FaceClassifier;
    private classifier: KNNClassifier;
    private labels: {};
    private constructor() {
        this.classifier = CreateClassifier();
    }
    static async getInstance() {
        if (!FaceClassifier.instance) {
            FaceClassifier.instance = new FaceClassifier();
            await FaceClassifier.instance.load('knn_classifier');
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
        return this.loadWeights(path.join(utils.MODEL_DIR, modelName))
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
            if (labels[data.personname] != undefined) {
                labels[data.personname] = clasNum;
                clasNum++;
            }
        }
        await dataList.forEach((data) => {
            this.classifier.addExample(tf.tensor(Float64Array.from(data.embedding)), labels[data.personname]);
        });
        this.labels = _.invert(labels);
        return await true;
    }

    async predict(embedding: Float64Array) {
        let res = await this.classifier.predictClass(tf.tensor(embedding), MAX_CLASSIFICATIONS);
        return res;
    }

    getNumClasses() {
        return this.classifier.getNumClasses();
    }
}