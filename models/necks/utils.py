import numpy as np


def bbox_jitter(bbox, num, delta):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    if num == 1:
        jitter = np.random.uniform(-delta, delta, 4)
        bboxes = [[max(bbox[0] + jitter[0] * w, 0.), min(bbox[1] + jitter[1] * h, 1.),
                   max(bbox[2] + jitter[2] * w, 0.), min(bbox[3] + jitter[3] * h, 1.)]]
                   
        return bboxes
    
    bboxes = [bbox]
    jitter = np.random.uniform(-delta, delta, [num - 1, 4])
    for i in range(num - 1):
        bboxes.append([max(bbox[0] + jitter[i][0] * w, 0.), min(bbox[1] + jitter[i][1] * h, 1.),
                       max(bbox[2] + jitter[i][2] * w, 0.), min(bbox[3] + jitter[i][3] * h, 1.)])
    return bboxes


def get_bbox_after_aug(aug_info, bbox, aug_threshold=0.3):
    if aug_info is None:
        return bbox
    
    cbox = aug_info['crop_box']
    w = cbox[2] - cbox[0]
    h = cbox[3] - cbox[1]
    
    l = max(min(bbox[0], cbox[2]), cbox[0])
    r = max(min(bbox[2], cbox[2]), cbox[0])
    t = max(min(bbox[1], cbox[3]), cbox[1])
    b = max(min(bbox[3], cbox[3]), cbox[1])
    
    if (b-t) * (r-l) <= (bbox[3]-bbox[1]) * (bbox[2]-bbox[0]) * aug_threshold:
        return None
    ret = [(l-cbox[0]) / w, (t-cbox[1]) / h, (r-cbox[0]) / w, (b-cbox[1]) / h]
    
    if aug_info['flip']:
        ret = [1. - ret[2], ret[1], 1. - ret[0], ret[3]]

    pad_ratio = aug_info['pad_ratio']
    ret = [ret[0] / pad_ratio[0], ret[1] / pad_ratio[1], ret[2] / pad_ratio[0], ret[3] / pad_ratio[1]]
    
    return ret
