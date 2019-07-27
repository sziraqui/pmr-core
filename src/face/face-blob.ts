import { Image, Rect } from 'nodoface';

export class FaceBlob {

    public faceImage: Image;
    public descriptor: Float32Array;
    public name: string;

    constructor(sourceImage: Image, public bbox: Rect, public confidence) {
        this.faceImage = sourceImage.extract(bbox);
        this.descriptor = null;
        this.name = null;
    }
}