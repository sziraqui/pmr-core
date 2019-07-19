import { SequenceCapture, ImageCapture, VideoCapture, Image } from "nodoface";
import { Facenet, utils } from "../";
import { FaceDetectorConfig, FaceDetector } from "../face/detection/main";
import { FaceRecogConfig } from "../face/recognition/recogniser-base";
import { ObjectBlob } from "./object-blob";
import { FaceBlob } from "../face";
import { Tensor3D } from "@tensorflow/tfjs-node";
import { IoUMap, IOU } from "./tracker-utils";
import { FaceMatcher } from "../face/recognition/macther";


export interface TrackerConfig {
    capture: SequenceCapture;//| ImageCapture | VideoCapture;
    detectorConfig?: FaceDetectorConfig;
    recogniserConfig?: FaceRecogConfig;
    inputFPS?: number;
    detectorFPS?: number;
    recogFPS?: number;
    checkInterval?: number;
    blobDelInterval?: number;
    iouThreshold?: number;
}

export class FaceTracker {

    capture: SequenceCapture; // | ImageCapture | VideoCapture;
    detector: FaceDetector;
    recogniser: Facenet;
    inputFPS: number;
    detectorFPS: number;
    recogFPS: number;
    checkInterval: number;
    blobDelInterval: number;
    iouThreshold: number;

    blobs: ObjectBlob[];
    recogConfig: FaceRecogConfig;
    detectorConfig: FaceDetectorConfig;

    private currFrame;
    private lastFrame;

    private faceMatcher: FaceMatcher;

    private resultArray;

    constructor(config: TrackerConfig, faceMatcher: FaceMatcher) {
        this.capture = config.capture;
        FaceDetector.getInstance(config.detectorConfig)
            .then(detector => this.detector = detector);
        Facenet.getInstance()
            .then(recog => this.recogniser = recog);
        this.inputFPS = config.inputFPS ? config.inputFPS : 30;
        this.detectorFPS = config.detectorFPS ? config.detectorFPS : 25;
        this.recogFPS = config.recogFPS ? config.recogFPS : 15;
        this.checkInterval = config.checkInterval ? config.checkInterval : 5;
        this.blobDelInterval = config.blobDelInterval ? config.blobDelInterval : 2;
        this.iouThreshold = config.iouThreshold ? config.iouThreshold : 0.75;

        this.recogConfig = config.recogniserConfig;

        this.faceMatcher = faceMatcher;

        this.resultArray = Array();
    }

    async next() {
        this.currFrame = this.nextFrame();
        if (this.currFrame == undefined) {
            return;
        }
        let faces = await this.detector.detect(this.currFrame);
        this.compareIOU(faces);

        this.faceMatcher.match(this.blobs, this.recogConfig.distanceThreshold);

        this.lastFrame = this.currFrame;

        return this.blobs;
    }

    initBlobs(faces: FaceBlob[]) {
        this.blobs = new Array<ObjectBlob>(faces.length);
        faces.forEach(async (face, i) => {
            this.blobs[i] = new ObjectBlob('face', face.bbox, this.capture.getFrameNumber(), face.confidence);
            faces[i].descriptor = await this.recogniser.embedding(utils.imageToTensor(face.faceImage) as Tensor3D)
            this.blobs[i].attrs['descriptor'] = faces[i].descriptor;
        });
    }

    async compareIOU(faces: FaceBlob[]) {
        if (this.blobs == undefined) {
            return this.initBlobs(faces);
        }
        let assigned = Array<boolean>(faces.length).fill(false);
        for (let i = 0; i < this.blobs.length; i++) {
            let closestIndex = -1;
            let closestIou = 0;
            for (let j = 0; j < faces.length; j++) {
                let iou = IOU(this.blobs[i].lastRect, faces[j].bbox);
                if (iou > this.iouThreshold &&
                    iou > closestIou &&
                    this.capture.getFrameNumber() - this.blobs[i].lastFrame < this.blobDelInterval &&
                    assigned[j] == false) {
                    closestIndex = j;
                    closestIou = iou;
                }
            }
            if (closestIndex != -1) {
                this.updateBlob(faces[closestIndex], i);
                assigned[closestIndex] = true;
            }
        }
        for (let i = 0; i < assigned.length; i++) {
            if (!assigned[i]) {
                if (faces[i].descriptor == undefined) {
                    faces[i].descriptor = await this.recogniser.embedding(utils.imageToTensor(faces[i].faceImage) as Tensor3D);
                }
                this.blobs.push(
                    new ObjectBlob('face', faces[i].bbox, this.capture.getFrameNumber(), faces[i].confidence, { descriptor: faces[i].descriptor })
                );
            }
        }

    }

    updateBlob(face: FaceBlob, targetIndex: number) {
        this.blobs[targetIndex].lastRect = face.bbox;
        this.blobs[targetIndex].lastFrame = this.capture.getFrameNumber();
        this.blobs[targetIndex].confidence = face.confidence;
        this.blobs[targetIndex].lastImage = face.faceImage;
        this.blobs[targetIndex].attrs['descriptor'] = face.descriptor;
    }

    track() {
        while (this.capture.isOpened() && this.capture.getProgress() < 1) {
            this.next();
        }
    }

    nextFrame() {
        let frame: Image;
        try {
            if (this.capture instanceof SequenceCapture) {
                frame = this.capture.getNextFrame()
            }
            // else if (this.capture instanceof ImageCapture) {
            //     frame = this.capture.getNextImage();
            // } else if (this.capture instanceof VideoCapture) {
            //     frame = this.capture.read();
            // }
        } catch (error) {
            console.log('Error loading next frame');
        }

        return frame;
    }
}