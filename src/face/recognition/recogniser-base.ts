import { FaceBlob } from '../face-blob';
import { Image } from 'nodoface';

export interface FaceRecogConfig {
    recogniser?: 'facenet' // | 'arcface'; // though arcface not working yet
    distanceThreshold?: number; // 1.15
    knownFaces?: FaceBlob[];
}

export interface FaceRecog {

    embedding(faceImage: Image): Promise<Float32Array>;

    distance(embedding1: Float32Array, embedding2: Float32Array): number;
}