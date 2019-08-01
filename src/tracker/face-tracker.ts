import { VideoCapture, ImageCapture, Image, SequenceCapture, showImage, waitKey } from "nodoface";
import { Facenet, utils } from "../";
import { FaceDetectorConfig, FaceDetector } from "../face/detection/main";
import { FaceRecogConfig } from "../face/recognition/recogniser-base";
import { ObjectBlob } from "./object-blob";
import { FaceBlob } from "../face";
import { Tensor3D, util } from "@tensorflow/tfjs-node";
import { IoUMap, IOU } from "./tracker-utils";
import { VisTracking } from './visualise';
import { FaceMatcher } from "../face/recognition/macther";


export interface TrackerConfig {
    capture: VideoCapture;
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

    capture: VideoCapture;
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

    private currFrame: Image;
    private lastFrame;

    private faceMatcher: FaceMatcher;

    private resultArray;

    private frameNo: number;

    constructor(config: TrackerConfig, detector: FaceDetector, recogniser: Facenet, faceMatcher: FaceMatcher) {
        this.capture = config.capture;
        this.detector = detector;
        this.recogniser = recogniser;
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

        this.frameNo = 0;
    }

    async track() {

        while (this.capture.isOpened()) {
            await this.next();
            VisTracking.annotate(this.currFrame, this.blobs)
        }

    }

    async next() {
        this.currFrame = this.nextFrame();

        let faces = await this.detector.detect(this.currFrame);
        console.log('Detections: ', faces.length);
        await this.assignBlobs(faces);
        console.log('Frame no.', this.frameNo);
        // this.lastFrame = this.currFrame;
        return await this.blobs;
    }

    nextFrame() {
        let frame: Image;
        try {
            if (this.capture instanceof VideoCapture) {
                frame = this.capture.read()
                this.frameNo++;
            }
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
        console.log('in assignBlobs')
        if (!this.blobs) {
            return await this.initBlobs(faces);
        }
        let faceAssigned = Array<boolean>(faces.length).fill(false);
        let blobAssigned = Array<boolean>(this.blobs.length).fill(false);
        // pass 1: Match detections by iou
        for (let i = 0; i < this.blobs.length; i++) {
            let closestIndex = -1;
            let closestIou = 0;
            for (let j = 0; j < faces.length; j++) {
                let iou = IOU(this.blobs[i].lastRect, faces[j].bbox);
                if ((iou > this.iouThreshold) &&
                    (iou > closestIou) &&
                    (this.frameNo - this.blobs[i].lastFrameNo < this.blobDelInterval) &&
                    (faceAssigned[j] == false)) {
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
                    faces[i].descriptor = await this.recogniser.embedding(faces[i].faceImage);
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
                if (index != -1) {
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
                    new ObjectBlob('face', faces[i].bbox, this.frameNo, faces[i].confidence, { descriptor: faces[i].descriptor })
                );
            }
        }

        this.deleteOldBlobs();
        // correct the rcognitions at every recogFPS interval
        if (this.frameNo % this.recogFPS == 0) {
            this.faceMatcher.match(this.blobs, this.recogConfig.distanceThreshold); // assign names from known names
        }
    }

    async initBlobs(faces: FaceBlob[]) {
        console.log('in initBlobs');
        this.blobs = new Array<ObjectBlob>();
        for (let i = 0; i < faces.length; i++) {
            let blob = new ObjectBlob('face', faces[i].bbox, this.frameNo, faces[i].confidence);
            faces[i].descriptor = await this.recogniser.embedding(faces[i].faceImage);
            blob.attrs['descriptor'] = faces[i].descriptor;
            this.blobs.push(blob);
        }
        await this.faceMatcher.match(this.blobs, this.recogConfig.distanceThreshold); // assign names from known names
    }

    updateBlob(face: FaceBlob, targetIndex: number) {
        console.log('in updateBlob: updated blob no.', targetIndex);
        this.blobs[targetIndex].lastRect = face.bbox;
        this.blobs[targetIndex].lastFrameNo = this.frameNo;
        this.blobs[targetIndex].confidence = face.confidence;
        this.blobs[targetIndex].lastImage = face.faceImage;
        this.blobs[targetIndex].attrs['descriptor'] = face.descriptor;
    }

    // delete blobs which are not seen since past blobDelInterval number of frames
    deleteOldBlobs() {
        let newBlobs = Array<ObjectBlob>();
        this.blobs.forEach((blob, i) => {
            if (blob.lastFrameNo - this.frameNo < this.blobDelInterval) {
                newBlobs.push(blob);
            }
        });
        this.blobs = newBlobs;
    }
}