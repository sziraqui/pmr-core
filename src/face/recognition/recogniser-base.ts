import { Tensor3D } from '@tensorflow/tfjs-node';
export interface FaceRecog {

    embedding(faceImage: Tensor3D): Promise<Float32Array>;

    distance(embedding1: Float32Array, embedding2: Float32Array): number;
}