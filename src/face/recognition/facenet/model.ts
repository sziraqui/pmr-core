import { FaceRecog } from "../recogniser-base";
import { FaceRecognitionNet } from 'face-api.js';
import * as path from "path";
import * as tf from '@tensorflow/tfjs-node';
import { Utils } from '../../../utils';

export class Facenet implements FaceRecog {

    public static weightsPath = path.join(Utils.MODEL_DIR, 'facenet_weights');

    private static instance: Facenet;

    private constructor(private model: FaceRecognitionNet) {
        this.model = model;
    }

    public static async getInstance() {
        if (Facenet.instance == undefined) {
            const model = new FaceRecognitionNet();
            await model.loadFromDisk(this.weightsPath);
            Facenet.instance = new Facenet(model);
        }
        return Facenet.instance;
    }

    async embedding(faceImage: tf.Tensor3D) {

        let descriptor = await this.model.computeFaceDescriptor(faceImage);
        return (descriptor as Float32Array);
    }

    distance(embedding1: Float32Array, embedding2: Float32Array): any {
        return tf.norm(tf.sub(embedding1, embedding2), 'euclidean').dataSync()[0];
    }

    isLoaded() {
        return this.model.isLoaded;
    }
}