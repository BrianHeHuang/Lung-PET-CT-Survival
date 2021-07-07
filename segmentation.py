import numpy as np
# from scipy.misc import imresize

axis_sum = {
    0: (1, 1),
    1: (0, 1),
    2: (0, 0),
}

axis_plane = {
    0: lambda image, plane : image[plane, :, :],
    1: lambda image, plane : image[:, plane, :],
    2: lambda image, plane : image[:, :, plane],
}

def calculate_percentile_slice(segmentation, percentile=100, axis=2):
    """
    Pass in 3D numpy array and return the index of the segmentation section at the requested percentile of size
    """
    i, j = axis_sum[axis]
    sum_on_plane = segmentation.sum(i).sum(j)
    sum_on_plane = sum_on_plane.astype('float')
    sum_on_plane[sum_on_plane == 0] = np.nan # this is necessary if we want to ignore all empty segmentation slices
    plane = np.where(sum_on_plane==np.nanpercentile(sum_on_plane, percentile, interpolation='nearest'))[0][0]
    return plane

def select_slice(image, segmentation, plane, axis=2):
    image = axis_plane[axis](image, plane)
    segmentation = axis_plane[axis](segmentation, plane)
    return image, segmentation

def bounding_box(segmentation):
    """
    Pass in 2D numpy array and get a smaller array around bounding box only
    """
    a = np.where(segmentation > 0)
    for i in reversed(range(5)):
        bounds = np.min(a[0]) - i, np.max(a[0]) + i, np.min(a[1]) - i, np.max(a[1] + i)
        okay_boundary = True
        for x in bounds:
            if x<0:
                okay_boundary = False
        if okay_boundary == True:
            return bounds
    return bounds

def crop(image, segmentation, bounds):
    cropped_image = image[bounds[0]: bounds[1], bounds[2]:bounds[3]]
    cropped_segmentation = segmentation[bounds[0]: bounds[1], bounds[2]:bounds[3]]
    return cropped_image, cropped_segmentation

def resize(image, size):
    return imresize(image, size, interp="bilinear")

def calculate_length(vector):
    out = 0
    for v in vector:
        out += float(v) ** 2
    return np.sqrt(out)

def calculate_voxel_size(vectors):
    out = 1
    for vector in vectors:
        if vector == "none":
            continue
        out *= calculate_length(vector)
    return out

def calculate_volume(array, metadata, filename):
    array[array > 0] = 1
    unit_volume = array.sum()
    space_directions = metadata.get('space directions')
    if space_directions is None:
        print("no space directions for {}".format(filename))
        return unit_volume
    voxel_size = calculate_voxel_size(space_directions)
    return unit_volume * voxel_size

