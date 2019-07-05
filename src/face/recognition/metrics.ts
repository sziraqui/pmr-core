import * as tf from '@tensorflow/tfjs-node';
import { TypedArray } from '@tensorflow/tfjs-core/dist/types';

/**
 * arcface loss ported from https://github.com/auroua/InsightFace_TF/blob/master/losses/face_losses.py#L5
 * @param embeddings 512-D face embeddings 
 * @param labels labels array of shape [batchSize, 1]
 * @param outClass output class number
 * @param s scalar value
 * @param m margin value
 */
export function arcface_loss(embeddings: tf.Tensor | TypedArray, labels: number[][], outClass: number, s: number = 64, m = 0.5): tf.Tensor {
    const cosM = Math.cos(m);
    const sinM = Math.sin(m);
    const mm = sinM * m;
    const threshold = Math.cos(Math.PI - m);

    const embeddingsNorm = tf.norm(embeddings, "euclidean", 1, true);
    embeddings = tf.div(embeddings, embeddingsNorm);
    let weights = weightInit([embeddings.shape[embeddings.shape.length - 1], outClass], "float32");
    let weightsNorm = tf.norm(weights, "euclidean", 0, true);
    weights = tf.div(weights, weightsNorm);

    const cosT = tf.matMul(embeddings, weights);
    const cosT2 = tf.square(cosT);
    const sinT2 = tf.sub(1, cosT2);
    const sinT = tf.sqrt(sinT2);
    const cosMt = tf.mul(s, tf.sub(tf.mul(cosT, cosM), tf.mul(sinT, sinM)));

    const condV = tf.sub(cosT, threshold);
    const cond = tf.cast(tf.relu(condV), "bool");

    const keepVal = tf.mul(s, tf.sub(cosT, mm));
    const costMtTemp = tf.where(cond, cosMt, keepVal);

    const mask = tf.oneHot(labels, outClass);
    const invMask = tf.sub(1, mask);

    const scalarCosT = tf.mul(s, cosT);
    const output = tf.add(tf.mul(scalarCosT, invMask), tf.mul(costMtTemp, mask));
    return output;
}

function weightInit(shape: number[], dtype: 'float32' | 'int32'): tf.Tensor {
    return tf.randomNormal(shape, 0, 1, dtype);
}
