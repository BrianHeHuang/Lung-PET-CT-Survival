from config import config

IMAGE = "imagingVolume.nrrd"

#manually segmented masks
MANUAL_SEGMENTATION = "segMask_GTV.nrrd"

#automatically segmented masks
AUTO_SEGMENTATION = "segMask_pred.nrrd"

T1 = config.MODE
#T2 = "T2"

ACCEPTED_FILENAMES = [
    IMAGE,
    MANUAL_SEGMENTATION,
    AUTO_SEGMENTATION
]
