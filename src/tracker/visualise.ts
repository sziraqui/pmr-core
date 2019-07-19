import { ObjectBlob } from '../tracker/object-blob';
import { drawRect, showImage, Image } from 'nodoface';

export const colorPallete = [
    [255, 0, 0], // red
    [255, 128, 0], // orange
    [255, 255, 0], // yellow
    [0, 255, 0], // green
    [0, 255, 255], // cyan
    [0, 0, 255], // blue
    [127, 0, 255], // purple
    [255, 0, 255], // pink
    [255, 0, 127], // pink-red
    [128, 128, 128], // gray
    [51, 102, 0], // dark green
    [102, 51, 0], // brown
    [255, 255, 255], // white
    [0, 0, 0] // black
];

export const colorNames = [
    'red',
    'orange',
    'yellow',
    'green',
    'cyan',
    'blue',
    'pink',
    'pink-red',
    'gray',
    'dark-green',
    'brown',
    'white',
    'black'
];

export function randomColor() {
    let index = Math.round(Math.random() * (colorPallete.length - 1));
    return colorPallete[index];
}

export class VisTracking {

    static annotate(image: Image, blobs: ObjectBlob[]) {
        blobs.forEach((blob, i) => {
            let color = colorPallete[blob.getId() % colorPallete.length];
            drawRect(image, blob.lastRect, color);
        });
    }
}