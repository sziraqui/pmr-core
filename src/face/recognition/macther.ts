import { ObjectBlob } from '../../tracker/object-blob';
import { Facenet } from './facenet/model';
export class FaceMatcher {

    constructor(private labels: string[], private descriptors: Float32Array[]) {
    }

    async match(blobs: ObjectBlob[], distThresh: number) {
        let facenet = await Facenet.getInstance();
        let assigned = Array(this.descriptors.length);
        for (let i = 0; i < blobs.length; i++) {
            let minDist = 100000000000;
            let closestIndex = -1;
            for (let j = 0; j < this.descriptors.length; j++) {
                let d = facenet.distance(this.descriptors[j], blobs[i]['descriptor']);
                if (d < minDist && !assigned[j] && d < distThresh) {
                    minDist = d;
                    closestIndex = j;
                }
            }
            if (closestIndex != -1) {
                blobs[i]['name'] = this.labels[closestIndex];
                blobs[i]['distance'] = minDist;
                assigned[closestIndex] = 1;
            }
        }
        return blobs; // now contains name
    }
}