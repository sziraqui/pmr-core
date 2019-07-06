import { describe } from 'mocha';
import { expect } from 'chai';
import { arcface_loss } from "./metrics";
import * as tf from '@tensorflow/tfjs-node';

describe('Metrics', () => {
    it('arcface_loss() execution', () => {
        const batchSize = 4;
        const embeddingLen = 2;
        const embeddings = tf.randomNormal([batchSize, embeddingLen], 0, 1, 'float32');
        const labels = [[0], [1], [0], [1]];
        const output = arcface_loss(embeddings, labels, 2);
        // output.print();
        expect(output.shape).to.eql([batchSize, batchSize, embeddingLen]);
    });
});
