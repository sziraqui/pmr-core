import { Image } from 'nodoface';
import { Tensor3D } from '@tensorflow/tfjs-node';
export interface FaceRecog {

    embedding(faceImage: Tensor3D); //: number[];

    distance(embedding1: number[], embedding2: number[]): number;
}