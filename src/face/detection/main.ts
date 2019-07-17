import { FaceDetectorHOG, FaceDetectorHaar, FaceDetectorMTCNN, Image, DetectionResult, drawRect } from "nodoface";
import { FaceBlob } from '../face-blob';

export interface FaceDetectorConfig {
    detector?: 'mtcnn' | 'haar' | 'hog';
    minConfidence?: number;
    drawBBox?: boolean;
    bboxColor?: [number, number, number];
    weightsPath?: string;
}

export class FaceDetector {

    private model;

    private static instance: FaceDetector;

    detector: 'mtcnn' | 'haar' | 'hog';
    minConfidence: number;
    drawBBox: boolean;
    bboxColor: [number, number, number];

    private constructor(config: FaceDetectorConfig) {
        this.detector = config.detector ? config.detector : 'mtcnn';
        this.minConfidence = config.minConfidence ? config.minConfidence : 0.6;
        this.drawBBox = config.drawBBox ? config.drawBBox : true;
        this.bboxColor = config.bboxColor ? config.bboxColor : [0, 255, 0];

    }

    public static async getInstance(config: FaceDetectorConfig = {}) {
        if (FaceDetector.instance == undefined) {
            FaceDetector.instance = new FaceDetector(config);

            if (FaceDetector.instance.detector == 'mtcnn') {
                FaceDetector.instance.model = new FaceDetectorMTCNN();
            } else if (FaceDetector.instance.detector == 'haar') {
                FaceDetector.instance.model = new FaceDetectorHaar();
            } else if (FaceDetector.instance.detector == 'hog') {
                FaceDetector.instance.model = new FaceDetectorHOG();
            }
            if (config.weightsPath) {
                await FaceDetector.instance.model.load(config.weightsPath);
            } else {
                console.log('W: Loading HOG face detector');
                await FaceDetector.instance.model.load();
            }
        }
        return FaceDetector.instance;
    }

    public async detect(image: Image) {
        let { detections: detections, confidences: confidences } = await this.model.detectFaces(image);
        let faces = Array<FaceBlob>();
        for (let i = 0; i < detections.length; i++) {
            if (confidences[i] > this.minConfidence) {
                faces.push(new FaceBlob(image, detections[i], confidences[i]));
                if (this.drawBBox) {
                    drawRect(image, detections[i], this.bboxColor);
                }
            }
        }
        return faces;
    }

}