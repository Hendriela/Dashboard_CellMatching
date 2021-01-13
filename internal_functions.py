import numpy as np
from skimage import measure
from scipy import ndimage
import math
import cv2 as cv
from skimage.registration import phase_cross_correlation
from scipy.ndimage import zoom
from scipy import spatial
from skimage.draw import polygon
import bisect
from skimage.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

offset = 10

# **********************************************************************************************************************
#               FORMATTING AND PLOTTING FUNCTIONS
# **********************************************************************************************************************

def format_footprints(session_npy_data):
    """ Formats the footprint data into a (num_neurons x dim x dim) format.

    Args:
        session_npy_data: npy file of session data (obtained by running prepare_data_dashboard.py)

    Returns:
        footprints (3D np.array): Array of neuron footprints (num_neurons x dim x dim)
    """
    num_neurons = session_npy_data['dff_trace'].shape[0]
    footprints = np.reshape(session_npy_data['spatial_masks'].toarray(),
                            (session_npy_data['template'].shape[0], session_npy_data['template'].shape[1], num_neurons),
                            order='F')
    footprints = np.transpose(footprints, (2, 0, 1))
    return footprints


def format_fig(matrix, title="No session uploaded", zoom=False, zoom_ratio=0.4, center_coords_x=100, center_coords_y=100,
               is_tiff_mode=False):
    """ Returns a styled plot of the given matrix """
    fig = px.imshow(matrix, color_continuous_scale=
    [[0.0, '#0d0887'],
     [0.0333333333333333, '#46039f'],
     [0.0444444444444444, '#7201a8'],
     [0.0555555555555555, '#9c179e'],
     [0.0666666666666666, '#bd3786'],
     [0.0888888888888888, '#d8576b'],
     [0.1111111111111111, '#ed7953'],
     [0.1333333333333333, '#fb9f3a'],
     [0.1444444444444444, '#fdca26'],
     [0.1777777777777777, '#f0f921'],
     [0.25, "white"], [0.4, "white"], [0.4, "grey"], [0.5, "grey"],
     [0.5, "red"], [0.6, "red"], [0.6, "green"], [0.7, "green"], [0.7, "pink"], [0.8, "pink"], [0.8, "black"],
     [1, "black"]], range_color=[0, 5])
    if is_tiff_mode:
        matrix = np.interp(matrix, (matrix.min(), matrix.max()), (0, 1))
        fig = px.imshow(matrix, zmin=0, zmax=1)
        # fig=px.imshow(matrix, color_continuous_scale='gray', zmin=matrix.min(), zmax=matrix.max())
        fig.add_trace(go.Scatter(x=[center_coords_x], y=[center_coords_y], marker=dict(color='red', size=5)))
    # fig_neuron = px.imshow(footprints_1[0])
    fig.update_layout(   coloraxis_showscale=False,
        autosize=False,
        width=350, height=350,
        margin=dict(l=5, r=5, t=25, b=5),
        title={"text": title,
               "yref": "paper",
               "xref": "paper",
               },
    )
    fig.update_traces(hoverinfo='none', hovertemplate=None)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    if zoom and zoom_ratio != 1:
        zoom_radius = matrix.shape[0]*zoom_ratio*0.5
        xmin = max(center_coords_x - zoom_radius , 0)
        xmax = min(matrix.shape[0], center_coords_x + zoom_radius )
        ymin = min(matrix.shape[1], center_coords_y + zoom_radius)
        ymax = max(center_coords_y - zoom_radius , 0)
        print(matrix.shape, center_coords_x, center_coords_y)
        fig.update_xaxes(range=[xmin, xmax], autorange=False)
        fig.update_yaxes(range=[ymin, ymax], autorange=False)
    return fig


# **********************************************************************************************************************
#               BASIC FUNCTIONS
# **********************************************************************************************************************

def pixel_neuron_ownership(footprint):
    """ Assigns an ownership neuron to each "pixel" of a 2D matrix (dim x dim).
        (If no neuron owns the pixel, the entry for pixel is -1.)

    Args:
        footprint (3D np.array): Array of neuron footprints (num_neurons x dim x dim)

    Returns:
        2D np.array: Array that represents pixel ownership
    """
    max_vals = np.amax(footprint, axis=0)
    pixel_owners = np.argmax(footprint, axis=0)
    pixel_owners = np.where(max_vals > 0.005, pixel_owners, -1)
    return pixel_owners


def compute_CoMs(footprints):
    """ Computes the center of mass of the neuron footprints.

    Args:
        footprints (3D np.array): Array of neuron footprints (num_neurons x dim x dim)

    Returns:
        2D array of shape (n_neurons, 2): Center of mass of neurons.

    """
    CoMs = np.full(footprints.shape[0], np.nan, dtype='f,f')
    for idx, footprint in enumerate(footprints):
        CoMs[idx] = ndimage.measurements.center_of_mass(footprint)
    return CoMs


def piecewise_fov_shift(ref_img, tar_img, n_patch=8):
    """ Calculates FOV-shift map between a reference and a target image. Images are split in n_patch X n_patch patches,
    and shift is calculated for each patch separately with phase correlation. The resulting shift map is scaled up and
    missing values interpolated to ref_img size to get an estimated shift value for each pixel.

    Args:
        ref_img (np.array): reference image
        tar_img (np.array): target image to which FOV shift is calculated. Has to be same dimensions as ref_img
        n_patch (int): root number of patches the FOV should be subdivided into for piecewise phase correlation

    Returns:
         np.array: Array containing estimated shifts per pixel (upscaled x_shift_map)
         np.array: Array containing estimated shifts per pixel (upscaled y_shift_map)
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


def shift_CoMs(CoMs, shift, dims):
    """ Shifts a center-of-mass coordinate point by a certain step size. Caps coordinates at 0 and dims limits
    Args:
        CoMs (iterable): X and Y coordinates of the center of mass.
        shift (iterable): amount by which to shift com, has to be same length as com.
        dims (iterable): dimensions of the FOV, has to be same length as com.
    Returns:
        shifted com
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



# **********************************************************************************************************************
#               FEATURE COMPUTATION
# **********************************************************************************************************************

def get_neuron_crop(footprint):
    """Computes minimal rectangle crop of the neuron footprint.

    Args:
        footprint (2D np.array): Footprint of a neuron

    Returns:
        2D array: Crop of the neuron footprint
    """
    coords_non0 = np.argwhere(footprint)
    x_min, y_min = coords_non0.min(axis=0)
    x_max, y_max = coords_non0.max(axis=0)
    # x_length_neuron_crop = x_max - x_min; y_length_neuron_crop = y_max - y_min
    cropped = footprint[x_min:x_max + 1, y_min:y_max + 1]
    return cropped


def get_area_crop(template, CoM, margin=50):
    """ Crops the image of the FoV to an area around the given point (CoM). Caps coordinates at 0 and dims limits.

    Args:
        template (2D array): FoV image
        CoM: Center of mass (x and y coordinate) of neurons.
        margin: Margin around center of mass to crop the FoV to.

    Returns:
        2D array: crop of FoV image
    """
    ymin = max(0, int(CoM[1] - margin))
    ymax = min(template.shape[1], int(CoM[1] + margin))
    xmin = max(0, int(CoM[0] - margin))
    xmax = min(template.shape[0], int(CoM[0] + margin))
    cropped = template[xmin:xmax, ymin:ymax]
    return cropped, (CoM[1] - ymin, CoM[0] - xmin)


def binarize_contour_area(footprint):
    """ Computes contour of neuron footprint and binarizes the resulting matrix

    Args:
        footprint(2D np.array): Footprint of a neuron

    Returns:
        2D np.array: Binarized, contoured Footprint of a neuron
    """
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

def neighbour_quadrant_split(coms, neuron_ref, neuron_idxs):
    """ Calculates the number of neurons that are in each quadrant of the reference neuron.

    Args:
        coms (2D array of shape (n_neurons, 2)): Center of mass of neurons.
        neuron_ref: Idx of the refernce neuron.
        neuron_idxs: Idxs of the neuron that shoudl be sorted into quadrants (with respect to the ref neuron)

    Returns:
        list (len=4): Number of neurons (of the given neuron idxs) in each quadrant of the neuron.
    """
    quadrant_split = [0, 0, 0, 0]
    x_ref = coms[neuron_ref][0]; y_ref = coms[neuron_ref][1]
    for idx in neuron_idxs:
        x = coms[idx][0]; y = coms[idx][1]
        if x_ref <= x and y_ref <= y:
            quadrant_split[0] += 1
        elif x_ref <= x and y_ref >= y:
            quadrant_split[1] += 1
        elif x_ref >= x and y_ref >= y:
            quadrant_split[2] += 1
        else:
            quadrant_split[3] += 1
    return quadrant_split


def compute_neuron_features(footprints, coms, template):
    """ Computes the neuron features (cropped neuron, cropped area, closest neuron angle, number of neurons in radius,
        number of neurons area quadrants) of all neurons of the session. These neuron features are used
        to calculate the features of the neuron pairs (one neuron from the ref and one neuron form the other session),
        which are used as input for the classifier.

    Args:
        footprints (3D np.array): Array of neuron footprints (num_neurons x dim x dim)
        coms (2D array of shape (n_neurons, 2)): Center of mass of neurons.
        template (2D array): FoV image

    Returns:
        2D array (n_neurons, n_features): List containing a list of features for each neuron.
    """
    features_list = []
    num_neurons = coms.shape[0]
    for neuron_idx in range(num_neurons):
        com = coms[neuron_idx]
        cropped_image_bin = binarize_contour_area(get_neuron_crop(footprints[neuron_idx]))
        area_image = get_area_crop(template, com)[0]
        coms_list = [list(com) for com in coms]
        distance, index = spatial.KDTree(coms_list).query(coms_list[neuron_idx], k=15) # return 15 closest neurons
        closest_idx = index[1]
        closest_neur_angle = math.atan2(com[1] - coms[closest_idx][1],
                                        com[0] - coms[closest_idx][0])
        num_neurons_in_radius = bisect.bisect(distance, 50)
        index_in_radius = index[: max(0, num_neurons_in_radius)]
        neighbours_quadrants = neighbour_quadrant_split(coms, neuron_idx, index_in_radius)
        neuron_features = {'com' : com, 'neur_image' : cropped_image_bin, 'area_image' : area_image,
                           'closest_neur_idx' : closest_idx, 'closest_neur_angle' : closest_neur_angle,
                          'num_neur_in_radius': num_neurons_in_radius, 'neighbour_quadrant_split': neighbours_quadrants}
        features_list.append(neuron_features)
    return features_list

def match_neurons_to_ref(footprints_ref, footprints_other,
                         coms_ref, coms_other,
                         template_ref, template_other):
    """ For each neuron of the reference session the classifier input features (dist, cropped neuron similarity,
        cropped area similarity, difference of the closest neuron angle, difference in number of neurons in radius,
        difference in number of neurons area quadrants) of each neuron in the non-reference session are computed and
        ordered by distance to the reference neuron.
     ! -> the features are computed for each possible neuron-pair (of the ref and other session)

    Args:
        footprints_ref (3D np.array): Array of neuron footprints (num_neurons x dim x dim) of reference session
        footprints_other (3D np.array): Array of neuron footprints (num_neurons x dim x dim) of other session
        coms_ref (2D array of shape (n_neurons, 2)): Center of mass of neurons of reference session
        coms_other (2D array of shape (n_neurons, 2)): Center of mass of neurons of other session
        template_ref (2D array): FoV image of reference session
        template_other (2D array): FoV image of other session

    Returns:
        array-like: List of neurons ranked in matching_ranked
        array_like:  matching_with_feature_info
    """
    num_neurons_ref = coms_ref.shape[0]
    num_neurons_other = coms_other.shape[0]
    matching_ranked = [None] * num_neurons_ref
    matching_with_feature_info = [None] * num_neurons_ref

    # compute shifted coms
    fov_shift = piecewise_fov_shift(template_ref, template_other)
    coms_shifted_other = shift_CoMs(coms_other, fov_shift, template_other.shape)

    # precompute features of each neuron of the neurons of the ref and other session
    print("compute ref features")
    features_ref = compute_neuron_features(footprints_ref, coms_ref, template_ref)
    print("compute other features")
    features_other = compute_neuron_features(footprints_other, coms_shifted_other, template_other)


    for neuron_idx_ref in range(num_neurons_ref):
        neur_features_ref = features_ref[neuron_idx_ref]
        curr_matches = []
        for neuron_idx_other in range(num_neurons_other):
            neur_features_other = features_other[neuron_idx_other]

            # FEATURE 1
            dist = math.sqrt((neur_features_ref['com'][0] - neur_features_other['com'][0]) ** 2
                             + (neur_features_ref['com'][1] - neur_features_other['com'][1]) ** 2)

            # FEATURE 2. cropped neuron similarity
            similarity_neuroncrop_score = cv.matchShapes(neur_features_ref['neur_image'],
                                                         neur_features_other['neur_image'],
                                                         1, 0.0)
            # FEATURE 3. cropped area similarity
            min_length_x = min(neur_features_ref['area_image'].shape[0], neur_features_other['area_image'].shape[0])
            min_length_y = min(neur_features_ref['area_image'].shape[1], neur_features_other['area_image'].shape[1])
            similarity_areacrop_score = mean_squared_error(neur_features_ref['area_image'][0:min_length_x, 0:min_length_y],
                                                        neur_features_other['area_image'][0:min_length_x, 0:min_length_y])

            # FEATURE 4. closest neuron angle
            angle_diff_closest_neur = abs(neur_features_other['closest_neur_angle']
                                          - neur_features_ref['closest_neur_angle'])

            # FEATURE 5. number of neurons in radius
            num_neurons_in_radius_diff = abs(neur_features_ref['num_neur_in_radius']
                                             - neur_features_other['num_neur_in_radius'])

            # FEATURE 6. number of neurons in radius in quadrants
            neighbour_quadrant_split_diff = [abs(a - b) for a, b in zip(neur_features_ref['neighbour_quadrant_split'],
                                            neur_features_other['neighbour_quadrant_split'])]

            curr_matches.append([neuron_idx_other, neur_features_other['com'][0], neur_features_other['com'][1],
                                 dist, similarity_neuroncrop_score, similarity_areacrop_score, angle_diff_closest_neur,
                                 num_neurons_in_radius_diff, *neighbour_quadrant_split_diff])

        # sorted_matches = sorted(curr_matches, key=Comp)
        sorted_matches = sorted(curr_matches, key=lambda x: x[3])
        matching_ranked[neuron_idx_ref] = [match_tuples[0] for match_tuples in sorted_matches]
        matching_with_feature_info[neuron_idx_ref] = sorted_matches[:20]
    return matching_ranked, matching_with_feature_info

