import { SequenceCapture, ImageCapture, FaceDetectorHaar, FaceDetectorHOG, FaceDetectorMTCNN } from "nodoface";
import { Facenet } from "../";

export interface TrackerConfig {
    capture: SequenceCapture | ImageCapture;
    detector: FaceDetectorHaar | FaceDetectorHOG | FaceDetectorMTCNN;
    recogniser: Facenet;
    inputFPS: number;
    procFPS: number;
    recogFPS: number;
}