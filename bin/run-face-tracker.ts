#!/usr/bin/env ts-node

import { Facenet, FaceDetector, FaceModelParameters, VideoCapture, FaceTracker, TrackerConfig, FaceBlob, utils } from "../";
import { FaceDetectorConfig } from "../src/face/detection/main";
import { FaceRecogConfig } from "../src/face/recognition/recogniser-base";
import { ArgumentParser } from 'argparse'
import { Image, readImage, DetectionResult } from 'nodoface';
import { Tensor3D } from "@tensorflow/tfjs";
import { FaceMatcher } from "../src/face/recognition/macther";

async function run(args) {

    console.log(args);

    const modelParams = new FaceModelParameters(process.argv);

    let detectorConfig: FaceDetectorConfig = { detector: 'mtcnn', weightsPath: modelParams.getMtcnnLocation(), drawBBox: false, minConfidence: 0.6 };
    let recogConfig: FaceRecogConfig = { recogniser: 'facenet', distanceThreshold: args.distanceThreshold };

    const [facenet, detector] = await Promise.all([
        await Facenet.getInstance(),
        await FaceDetector.getInstance(detectorConfig)
    ]);

    const capture: VideoCapture = new VideoCapture();
    capture.open(args.file);

    let images = Array<Image>();
    args.faces.forEach(async (file) => {
        images.push(await readImage(file));
    });

    let descriptors = Array<Float32Array>();

    for (let i = 0; i < images.length; i++) {
        let faceBlob: FaceBlob = (await detector.detect(images[i]))[0];
        descriptors.push(await facenet.embedding(utils.imageToTensor(faceBlob.faceImage) as Tensor3D));
    }

    let faceMatcher = new FaceMatcher(args.names, descriptors);

    let trackerConfig: TrackerConfig = { inputFPS: args.i_fps, iouThreshold: args.it, detectorFPS: args.d_fps, blobDelInterval: args.bdel, capture: capture, checkInterval: 5, detectorConfig: detectorConfig, recogniserConfig: recogConfig, recogFPS: args.r_fps };

    const tracker = new FaceTracker(trackerConfig, detector, facenet, faceMatcher);

    tracker.track();

}

let parser = new ArgumentParser();
parser.addArgument(['-file', '--file'], { help: 'Video file to process', required: false });
parser.addArgument(['-targetFaces', '--faces'], { nargs: '*', help: 'List of images each containing single face', required: false });
parser.addArgument(['-names', '--names'], { nargs: '*', help: 'List of labels corresponding to targetFaces in same order', required: true });
parser.addArgument(['-inputFPS', '--i-fps'], { help: 'Video input fps', defaultValue: 30, required: false, type: parseInt });
parser.addArgument(['-detectionFPS', '--d-fps'], { help: 'FPS at which to run detections', defaultValue: 30, required: false, type: parseInt });
parser.addArgument(['-recognitionFPS', '--r-fps'], { help: 'FPS at which to run recognition', defaultValue: 30, required: false, type: parseInt });
parser.addArgument(['-iouThreshold', '--it'], { help: 'Threshold for IOU based comparisions', defaultValue: 0.75, required: false, type: parseInt });
parser.addArgument(['-distanceThreshold', '--dt'], { help: 'Threshold for face matcher', defaultValue: 1.15, required: false, type: parseInt });
parser.addArgument(['-blobDelInterval', '--bdel'], { help: 'Frames till which to keep old blobs', type: parseInt, defaultValue: 3, required: false });

let args = parser.parseArgs();

run(args);