from config import config

IMAGE = "imagingVolume.nrrd"
SEGMENTATION = "segMask_GTV.nrrd"

T1 = config.MODE
#T2 = "T2"

ACCEPTED_FILENAMES = [
    IMAGE,
    SEGMENTATION,
]
