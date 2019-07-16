import { expect } from 'chai';
import { describe } from 'mocha';
import { Facenet } from './model';
import * as tf from '@tensorflow/tfjs-node';

describe('Facenet', () => {
    it('.getInstance()', async () => {
        const facenet = await Facenet.getInstance();
        expect(facenet.isLoaded()).to.equal(true);
    });
    it('.embedding()', async () => {
        const facenet = await Facenet.getInstance();
        let embedding = await facenet.embedding(tf.randomNormal([224, 224, 3], 0, 2, 'float32', 0));
        expect(embedding.length).to.equal(128);
    });

    it('.distance()', async () => {
        const facenet = await Facenet.getInstance();
        let embedding0 = await facenet.embedding(tf.randomNormal([224, 224, 3], 0, 1, 'float32', 0));
        let embedding1 = await facenet.embedding(tf.randomNormal([224, 224, 3], 0, 127, 'float32', 1));
        expect(embedding0.length).to.equal(128);
        expect(embedding1.length).to.equal(128);
        let distance = facenet.distance(embedding0, embedding1);
        console.log('Distance', distance);
        expect(distance).to.greaterThan(0);
    });

});