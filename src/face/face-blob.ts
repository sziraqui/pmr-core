import { Image, Rect } from 'nodoface';

export class FaceBlob {
    public faceImage: Image;
    constructor(sourceImage: Image, public bbox: Rect, public confidence) {
        // TODO: get faceImage
    }
}