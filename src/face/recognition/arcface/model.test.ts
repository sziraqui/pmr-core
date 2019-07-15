import { describe } from 'mocha';
import { expect } from 'chai';
import { ArcFace } from "./model";
import * as tf from '@tensorflow/tfjs-node';

describe('ArcFace pretrained', () => {
    it('.getInstance()', async () => {
        let arcFace = await ArcFace.getInstance();
        let inp = tf.randomNormal([1, 224, 224, 3], 0, 1, 'float32');
        let res = await arcFace.embedding(inp);
        /** 
         * Error: Input tensor count mismatch,the frozen model has 2 placeholders, while there are 1 input tensors. 
         * */
    });
})