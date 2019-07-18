import { SequenceCapture, ImageCapture, FaceDetectorHaar, FaceDetectorHOG, FaceDetectorMTCNN, VideoCapture } from "nodoface";
import { Facenet } from "../";
import { FaceDetectorConfig } from "../face/detection/main";
import { FaceRecogConfig } from "../face/recognition/recogniser-base";

export interface TrackerConfig {
    capture: SequenceCapture | ImageCapture | VideoCapture;
    detectorConfig: FaceDetectorConfig;
    recogniserConfig: FaceRecogConfig;
    inputFPS: number;
    procFPS: number;
    recogFPS: number;
}