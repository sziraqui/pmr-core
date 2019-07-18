import { Rect } from "nodoface";

export function IOU(rect1: Rect, rect2: Rect) {
    let area1 = rect1.height * rect1.width;
    let area2 = rect2.height * rect2.width;
    const width = Math.max(0.0, Math.min(rect1.x + rect1.width, rect2.x + rect2.width) - Math.max(rect1.x, rect2.x));
    const height = Math.max(0.0, Math.min(rect1.y + rect1.height, rect2.y + rect2.height) - Math.max(rect1.y, rect2.y))
    const interSection = width * height
    return interSection / (area1 + area2 - interSection);
}

export interface IoUMap {
    i: number;
    j: number;
    iou: number;
}