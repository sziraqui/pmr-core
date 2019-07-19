import { Rect, Image, utils } from "../";

export type BlobType = 'face' | 'object';

export class ObjectBlob {
    private type: BlobType;
    lastRect: Rect;
    private id: number;
    firstFrameNo: number;
    lastFrameNo: number;
    lastImage: Image;
    attrs: Object;
    confidence: number;
    private static count: number = 0;

    constructor(type: BlobType, bbox: Rect, frameNo: number, confidence: number, attrs?: Object) {
        this.type = type;
        this.firstFrameNo = this.lastFrameNo = frameNo;
        this.lastRect = bbox;
        this.attrs = attrs;
        this.confidence = confidence;
        this.id = ObjectBlob.count++;
    }

}