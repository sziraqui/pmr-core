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
        this.detectorConfig = this.detectorConfig;

        this.faceMatcher = faceMatcher;

        this.resultArray = Array();
    }

    async track() {

        while (this.capture.isOpened() && this.capture.getProgress() < 1) {
            await this.next();
        }
    }

    async next() {
        this.currFrame = this.nextFrame();
        if (this.currFrame == undefined) {
            return;
        }
        let faces = await this.detector.detect(this.currFrame);
        this.assignBlobs(faces);

        // delete blobs that are no longer in frame

        this.lastFrame = this.currFrame;
        return this.blobs;
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
    /**
     * Findinding which new face belongs to which old face
     * @param faces 
     */
    async assignBlobs(faces: FaceBlob[]) {
        if (this.blobs == undefined) {
            return this.initBlobs(faces);
        }
        let faceAssigned = Array<boolean>(faces.length).fill(false);
        let blobAssigned = Array<boolean>(this.blobs.length).fill(false);
        // pass 1: Match detections by iou
        for (let i = 0; i < this.blobs.length; i++) {
            let closestIndex = -1;
            let closestIou = 0;
            for (let j = 0; j < faces.length; j++) {
                let iou = IOU(this.blobs[i].lastRect, faces[j].bbox);
                if (iou > this.iouThreshold &&
                    iou > closestIou &&
                    this.capture.getFrameNumber() - this.blobs[i].lastFrameNo < this.blobDelInterval &&
                    faceAssigned[j] == false) {
                    closestIndex = j;
                    closestIou = iou;
                }
            }
            // update properties of blob with its matched detection
            if (closestIndex != -1) {
                this.updateBlob(faces[closestIndex], i); // update using IOU measure
                faceAssigned[closestIndex] = true;
                blobAssigned[i] = true;
            }
        }
        // pass 2: Match remaining blobs using distance between their face descriptors
        // Should be executed at recogFPS
        for (let i = 0; i < faceAssigned.length; i++) {
            if (!faceAssigned[i]) {
                // TODO: compute descriptors in parallel
                if (faces[i].descriptor == undefined) {
                    faces[i].descriptor = await this.recogniser.embedding(utils.imageToTensor(faces[i].faceImage) as Tensor3D);
                }
                let minDist = 1000000000000;
                let index = -1;
                for (let j = 0; j < this.blobs.length; j++) {
                    if (!blobAssigned[j]) {
                        let d = this.recogniser.distance(faces[i].descriptor, this.blobs[j].attrs['descriptor'] as Float32Array);
                        if (d < minDist) {
                            minDist = d;
                            index = j;
                        }
                    }
                }
                if (index! - 1) {
                    this.updateBlob(faces[i], index); // assign `i`th face of faces to blob `index`th blob of blobs
                    faceAssigned[i] = true;
                    blobAssigned[index] = true;
                }
            }
        }
        // Create new blob for faces that cannot be identified
        for (let i = 0; i < faceAssigned.length; i++) {
            if (!faceAssigned[i]) {
                this.blobs.push(
                    new ObjectBlob('face', faces[i].bbox, this.capture.getFrameNumber(), faces[i].confidence, { descriptor: faces[i].descriptor })
                );
            }
        }

        this.deleteOldBlobs();
        // correct the rcognitions at every recogFPS interval
        if (this.capture.getFrameNumber() % this.recogFPS == 0) {
            this.faceMatcher.match(this.blobs, this.recogConfig.distanceThreshold); // assign names from known names
        }
    }

    initBlobs(faces: FaceBlob[]) {
        this.blobs = new Array<ObjectBlob>(faces.length);
        faces.forEach(async (face, i) => {
            this.blobs[i] = new ObjectBlob('face', face.bbox, this.capture.getFrameNumber(), face.confidence);
            faces[i].descriptor = await this.recogniser.embedding(utils.imageToTensor(face.faceImage) as Tensor3D)
            this.blobs[i].attrs['descriptor'] = faces[i].descriptor;
        });
        this.faceMatcher.match(this.blobs, this.recogConfig.distanceThreshold); // assign names from known names
    }

    updateBlob(face: FaceBlob, targetIndex: number) {
        this.blobs[targetIndex].lastRect = face.bbox;
        this.blobs[targetIndex].lastFrameNo = this.capture.getFrameNumber();
        this.blobs[targetIndex].confidence = face.confidence;
        this.blobs[targetIndex].lastImage = face.faceImage;
        this.blobs[targetIndex].attrs['descriptor'] = face.descriptor;
    }

    // delete blobs which are not in past blobDelInterval number of frames
    deleteOldBlobs() {
        let newBlobs = Array<ObjectBlob>();
        this.blobs.forEach((blob, i) => {
            if (blob.lastFrameNo - this.capture.getFrameNumber() < this.blobDelInterval) {
                newBlobs.push(blob);
            }
        });
        this.blobs = newBlobs;
    }
}