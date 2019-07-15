import { FaceRecog } from "../recogniser-base";
import { Utils } from "../../../utils";
import * as tf from '@tensorflow/tfjs-node';

export class ArcFace implements FaceRecog {

    private model: tf.FrozenModel;

    private static recog: ArcFace;

    private static readonly MODEL_NAME = 'arcface_resnet_v1_50';

    private constructor(model: tf.FrozenModel) {
        this.model = model;
    }

    static async getInstance() {
        if (ArcFace.recog == undefined) {
            let model = await Utils.loadFrozenModel(this.MODEL_NAME);
            ArcFace.recog = new ArcFace(model);
        }
        return ArcFace.recog;
    }

    async embedding(faceImage: tf.Tensor) {
        let res = await this.model.predict(faceImage);
        return res;
    }

    distance(embedding1: number[], embedding2: number[]): number {
        throw new Error("Method not implemented.");
    }


}