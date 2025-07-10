import numpy as np

def is_box_in_roi(box, roi):
    return is_point_in_roi(box[:2], roi) and is_point_in_roi(box[2:], roi)


def is_point_in_roi(pt, rect):
    return rect[0] < pt[0] < rect[2] and rect[1] < pt[1] < rect[3]


def check_overlap(boxA, boxB):
    # boxA = config.ROI

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    overlap_percentage = interArea / boxBArea

    # return the intersection over union value
    return overlap_percentage


def remove_overlap_boxes(bounding_boxes, threshold=0.9):
    overlapping_boxes = []
    mask = np.ones(len(bounding_boxes), dtype=np.bool)
    for i in range(len(bounding_boxes)):
        for j in range(i + 1, len(bounding_boxes)):
            if mask[j]:
                overlap_percentage = check_overlap(
                    bounding_boxes[i], bounding_boxes[j])
                if overlap_percentage > threshold:
                    overlapping_boxes.append(bounding_boxes[j])
                    mask[j] = 0

    return overlapping_boxes, mask


def remove_box_out_of_roi(bounding_boxes, roi):
        in_roi_boxes = []
        mask = np.zeros(len(bounding_boxes), dtype=np.bool)
        for i, box in enumerate(bounding_boxes):
            x1, y1, x2, y2 = box
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            if is_point_in_roi((cX, cY), roi) is True:
                in_roi_boxes.append(box)
                mask[i] = 1
                
        return in_roi_boxes, mask


def get_track_id(object_tracker, box):
        # Get track id
        _track_id = 0
        startX, startY, endX, endY = box
        cX = int((startX + endX) / 2.0)
        cY = int((startY + endY) / 2.0)
        inputCentroid = (cX, cY)
        for track_id, centroid in object_tracker.items():
            # Check centroid is inside the top box
            if inputCentroid[0] == centroid[0] and inputCentroid[1] == centroid[1]:
                _track_id = track_id
                
        return _track_id
