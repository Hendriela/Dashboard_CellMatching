import numpy as np
from scipy import ndimage, signal
import plotly.express as px
import pandas as pd
# import matplotlib.pyplot as plt
from skimage import measure, data, io, filters
from scipy import ndimage, signal
import math
import cv2 as cv
from skimage.registration import phase_cross_correlation
from operator import itemgetter
from scipy.ndimage import zoom
from scipy import spatial
from skimage.draw import line, polygon, circle, ellipse

offset = 10


def pixel_neuron_ownership(footprint):
    """Assigns an ownershiop neuron to each "pixel" of a 2D (dim x dim). 
        if no neuron owns the pixel, the entry for pixel is -1

         Args:
             footprint:    np.ndarray (3D)
                           Matrix of neuron footprints (num_neurons x dim x dim)
    """
    max_vals = np.amax(footprint, axis=0)
    pixel_owners = np.argmax(footprint, axis=0)
    pixel_owners = np.where(max_vals > 0.005, pixel_owners, -1)
    return pixel_owners


def compute_CoMs(footprints):
    CoMs = np.full(footprints.shape[0], np.nan, dtype='f,f')
    for idx, footprint in enumerate(footprints):
        CoMs[idx] = ndimage.measurements.center_of_mass(footprint)
    return CoMs


def piecewise_fov_shift(ref_img, tar_img, n_patch=8):
    """
    Calculates FOV-shift map between a reference and a target image. Images are split in n_patch X n_patch patches, and
    shift is calculated for each patch separately with phase correlation. The resulting shift map is scaled up and
    missing values interpolated to ref_img size to get an estimated shift value for each pixel.
    :param ref_img: np.array, reference image
    :param tar_img: np.array, target image to which FOV shift is calculated. Has to be same dimensions as ref_img
    :param n_patch: int, root number of patches the FOV should be subdivided into for piecewise phase correlation
    :return: two np.arrays containing estimated shifts per pixel (upscaled x_shift_map, upscaled y_shift_map)
    """
    img_dim = ref_img.shape
    patch_size = int(img_dim[0] / n_patch)

    shift_map_x = np.zeros((n_patch, n_patch))
    shift_map_y = np.zeros((n_patch, n_patch))
    for row in range(n_patch):
        for col in range(n_patch):
            curr_ref_patch = ref_img[row * patch_size:row * patch_size + patch_size,
                             col * patch_size:col * patch_size + patch_size]
            curr_tar_patch = tar_img[row * patch_size:row * patch_size + patch_size,
                             col * patch_size:col * patch_size + patch_size]
            patch_shift = phase_cross_correlation(curr_ref_patch, curr_tar_patch, upsample_factor=100,
                                                  return_error=False)
            shift_map_x[row, col] = patch_shift[0]
            shift_map_y[row, col] = patch_shift[1]
    shift_map_x_big = zoom(shift_map_x, patch_size, order=3)
    shift_map_y_big = zoom(shift_map_y, patch_size, order=3)
    return shift_map_x_big, shift_map_y_big


def format_footprints(session_npy_data):
    num_neurons = session_npy_data['dff_trace'].shape[0]
    footprints = np.reshape(session_npy_data['spatial_masks'].toarray(),
                            (session_npy_data['template'].shape[0], session_npy_data['template'].shape[1], num_neurons),
                            order='F')
    footprints = np.transpose(footprints, (2, 0, 1))
    return footprints


def compute_CoMs(footprints):
    CoMs = np.full(footprints.shape[0], np.nan, dtype='f,f')
    for idx, footprint in enumerate(footprints):
        CoMs[idx] = ndimage.measurements.center_of_mass(footprint)
    return CoMs


def get_neuron_crop(footprint):
    coords_non0 = np.argwhere(footprint)
    x_min, y_min = coords_non0.min(axis=0);
    x_max, y_max = coords_non0.max(axis=0)
    # x_length_neuron_crop = x_max - x_min; y_length_neuron_crop = y_max - y_min
    cropped = footprint[x_min:x_max + 1, y_min:y_max + 1]
    return cropped


def get_area_crop(template, CoM, margin=50):
    ymin = max(0, int(CoM[1] - margin));
    ymax = min(template.shape[1], int(CoM[1] + margin));
    xmin = max(0, int(CoM[0] - margin));
    xmax = min(template.shape[0], int(CoM[0] + margin));
    print(CoM[0], CoM[1], template.shape[0], template.shape[1], xmin, xmax, ymin, ymax)
    cropped = template[xmin:xmax, ymin:ymax]
    print(template.shape[0] - ymax, template.shape[0] - ymin,
          template.shape[1] - xmax, template.shape[0] - xmin)
    return cropped, (CoM[1] - ymin, CoM[0] - xmin)  # (CoM[1]-template.shape[0]+ymax, CoM[0]-template.shape[1]+xmax)


def binarize_contour_area(footprint):
    contours = measure.find_contours(footprint, 0.05, fully_connected='high')
    # if no contour binarize footprint
    new_footprint = np.copy(footprint)
    if contours:
        rr, cc = polygon(contours[0][:, 0], contours[0][:, 1], footprint.shape)
        new_footprint[rr, cc] = 255
    else:
        print("no contour")
        new_footprint = np.where(footprint > 0.07, 1.0, 0.0)
    return new_footprint


def shift_CoMs(CoMs, shift, dims):
    """
    Shifts a center-of-mass coordinate point by a certain step size. Caps coordinates at 0 and dims limits
    :param com: iterable, X and Y coordinates of the center of mass
    :param shift: iterable, amount by which to shift com, has to be same length as com
    :param dims: iterable, dimensions of the FOV, has to be same length as com
    :return: shifted com
    """
    # shift the CoM by the given amount
    print(CoMs.shape)
    coms_shifted = np.full(CoMs.shape[0], np.nan, dtype='f,f')
    for idx, com in enumerate(CoMs):
        com_shift = [com[0] - shift[0][int(com[0])][int(com[1])], com[1] - shift[1][int(com[0])][int(com[1])]]
        # print(com, shift[idx], com_shift)
        # cap CoM at 0 and dims limits
        com_shift = [0 if x < 0 else x for x in com_shift]
        for coord in range(len(com_shift)):
            if com_shift[coord] > dims[coord]:
                com_shift[coord] = dims[coord]
        coms_shifted[idx] = tuple(com_shift)
    return coms_shifted

def getclosest_neighbour(coms, ref_neuron_idx):

    distance,index = spatial.KDTree(coms).query(ref_neuron_idx, k=2)
    print(distance, index)


# compare feature tuples!
class Comp(object):
    def __init__(self, tup):
        self.tup = tup

    def __lt__(self, other):
        # If the difference is less or equal the offset of the second item compare the third
        if abs(self.tup[1] - other.tup[1]) <= offset:
            # The lower the result of cv.matchShape, the better the match is.
            return self.tup[2] < other.tup[2]
        # otherwise compare them as usual
        else:
            return self.tup[1] < other.tup[1]


def match_neurons_to_ref(footprints_ref, footprints_other,
                         coms_ref, coms_other,
                         template_ref, template_other):
    num_neurons_ref = coms_ref.shape[0]
    num_neurons_other = coms_other.shape[0]
    fov_shift = piecewise_fov_shift(template_ref, template_other)
    coms_shifted_other = shift_CoMs(coms_other, fov_shift, template_other.shape)
    matching_ranked = [None] * num_neurons_ref
    matching_with_feature_info = [None] * num_neurons_ref
    for neuron_idx_ref in range(num_neurons_ref):
        com_ref = coms_ref[neuron_idx_ref]
        cropped_image_ref_bin = binarize_contour_area(get_neuron_crop(footprints_ref[neuron_idx_ref]))
        curr_matches = []
        for neuron_idx_other in range(num_neurons_other):
            com_other = coms_other[neuron_idx_other]
            cropped_image_other_bin = binarize_contour_area(get_neuron_crop(footprints_other[neuron_idx_other]))
            similarity_neuroncrop_score = cv.matchShapes(cropped_image_other_bin, cropped_image_ref_bin, 1, 0.0)
            # print(com_ref[0], com_ref[1], com_other[0], com_other[1])
            dist = math.sqrt((com_ref[0] - com_other[0]) ** 2 + (com_ref[1] - com_other[1]) ** 2)
            curr_matches.append((neuron_idx_other, dist, similarity_neuroncrop_score))
        sorted_matches = sorted(curr_matches, key=Comp)
        # print(sorted_matches)
        matching_ranked[neuron_idx_ref] = [match_tuples[0] for match_tuples in sorted_matches]
        matching_with_feature_info[neuron_idx_ref] = sorted_matches
        print(neuron_idx_other, neuron_idx_ref, num_neurons_other, num_neurons_ref)
    print(len(matching_ranked))
    return matching_ranked, matching_with_feature_info


def match_neurons_to_ref_old(footprints_ref, footprints_other,
                             coms_ref, coms_other,
                             template_ref, template_other):
    num_neurons_ref = coms_ref.shape[0];
    num_neurons_other = coms_other.shape[0]
    fov_shift = piecewise_fov_shift(template_ref, template_other)
    coms_shifted_other = shift_CoMs(coms_other, fov_shift, template_other.shape)
    matching_ranked = [None] * num_neurons_ref
    for neuron_idx_ref in range(num_neurons_ref):
        com_ref = coms_ref[neuron_idx_ref]
        curr_matches = []
        for neuron_idx_other in range(num_neurons_other):
            com_other = coms_other[neuron_idx_other]
            footprint_other = footprints_other[neuron_idx_other];
            # print(com_ref[0], com_ref[1], com_other[0], com_other[1])
            dist = math.sqrt((com_ref[0] - com_other[0]) ** 2 + (com_ref[1] - com_other[1]) ** 2)
            curr_matches.append((neuron_idx_other, dist))
        curr_matches.sort(key=itemgetter(1))
        matching_ranked[neuron_idx_ref] = [match_tuples[0] for match_tuples in curr_matches]
    print(len(matching_ranked))
    return matching_ranked


"""
def match_neurons_across_session(footprints: list, coms: list, templates: list):
    for i in range(1, len(footprints)):
        best_match_idx = match_neurons_across_session(footprints[0], footprints[i], coms[0], coms[i],template[0], template[i])





def match_neurons_across_session_debugging(session_ref, session_other):
    num_neurons_ref = session_ref['dff_trace'].shape[0]; num_neurons_other = session_other['dff_trace'].shape[0]
    footprints_ref = format_footprints(session_ref); footprints_other = format_footprints(session_other);
    CoMs_ref = compute_CoMs(footprints_ref); CoMs_other = compute_CoMs(footprints_other)
    template_ref = session_ref['template']; template_other = session_other['template']
    print(piecewise_fov_shift(template_ref, template_other)[0].shape)
    print(piecewise_fov_shift(template_ref, template_other)[1].shape)
    fov_shift = piecewise_fov_shift(template_ref, template_other)
    CoMs_shifted_other = shift_CoMs(CoMs_other, fov_shift, template_other.shape)

    for neuron_idx_ref in range(num_neurons_ref):
        footprint_ref = footprints_ref[neuron_idx_ref];  CoM_ref = CoMs_ref[neuron_idx_ref]
        cropped_image_ref = get_neuron_crop(footprint_ref)
        cropped_image_ref_bin = binarize_contour_area(cropped_image_ref)
        area_image_ref, new_CoM_ref = get_area_crop(template_ref, CoM_ref)
        possible_matches_idxs = []
        # find neurons with CoM in range
        for neuron_idx_other in range(num_neurons_other):
            CoM_other = CoMs_other[neuron_idx_other]
            footprint_other = footprints_other[neuron_idx_other];
            print(CoM_ref[0], CoM_ref[1], CoM_other[0], CoM_other[1])
            dist = math.sqrt((CoM_ref[0]-CoM_other[0])**2 + (CoM_ref[1]-CoM_other[1])**2)
            if dist < 40:
                possible_matches_idxs.append(neuron_idx_other)
                # get neuron crop similarity
                cropped_image_other = get_neuron_crop(footprint_other)
                cropped_image_other_bin = binarize_contour_area(cropped_image_other)
                similarity_neuroncrop_score = cv.matchShapes(cropped_image_other_bin, cropped_image_ref_bin, 1, 0.0)
                # get neuron area crop similarity
                area_image_other, new_CoM_other = get_area_crop(template_other, CoM_other)
                area_image_other, new_CoM_other = get_area_crop(template_other, CoM_other)
                fig, axs = plt.subplots(3, 1)
                axs[0].imshow(template_ref)
                print("com_ref ", CoM_ref[0], CoM_ref[1] )
                axs[0].scatter(CoM_ref[1], CoM_ref[0])
                axs[1].imshow(template_other)
                axs[1].scatter(CoM_other[1], CoM_other[0])
                axs[1].scatter(CoMs_shifted_other[neuron_idx_other][1], CoMs_shifted_other[neuron_idx_other][0])
                axs[2].imshow(area_image_other)
                axs[2].scatter(new_CoM_other[0],new_CoM_other[1])

                plt.show()

                #cropped_image_other_bin = binarize_contour_area(cropped_image_other)
                similarity_areacrop_score = cv.matchShapes(area_image_ref, cropped_image_other, 1, 0.0)

                # for neurons in range find similarity of neuron_crop and area_crop

                marked_footprint_ref  = np.where(footprint_ref>0.07, 100, template_ref)
                marked_footprint_other  = np.where(footprint_other>0.07, 100, template_other)

                ### plot to visualize piecewise_fov_shift


# debugging
UPLOAD_DIRECTORY = "C:\\Users\\annahs\\Documents\\Footprints\\alignment_session_data\\"

session_ref = np.load(UPLOAD_DIRECTORY + "20200819.npy", allow_pickle=True).item()
session_other = np.load(UPLOAD_DIRECTORY + "20200826.npy", allow_pickle=True).item()
footprints_ref = format_footprints(session_ref)
footprints_other = format_footprints(session_other)

result = match_neurons_to_ref(footprints_ref, footprints_other,
                             compute_CoMs(footprints_ref), compute_CoMs(footprints_other),
                             session_ref['template'], session_other['template'])
print(result[0])
"""
