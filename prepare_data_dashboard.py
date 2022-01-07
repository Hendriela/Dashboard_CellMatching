""" Extraction of the data needed for the dashboard from the h5py files from the caiman pipeline

Set 'input_folderpaths' to the filepaths, where you have stored the h5py file of your session data.
The folder must include:
    1) mean intensity image (named mean_intensity_image.tif) and
    2) the session data (named cnm_results.hdf5)

(Alternatively, set the flag to is_pkl_file = True and use the file "\pcf_results.pickle")
"""

import numpy as np
import pickle
import h5py
from scipy import sparse
from PIL import Image


# **********************************************************************************************************************
#               USER-SPECIFIC SETTINGS
# **********************************************************************************************************************

input_folderpaths = [r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M38\20200818",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M38\20200819",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M38\20200820",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M38\20200821",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M38\20200824",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M38\20200826",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M38\20200827",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M38\20200830",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M38\20200902",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M38\20200905",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M38\20200908",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\M38\20200911"]


# ! in same order as the inputfile paths
output_filepaths =  [r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\footprints\M38\20200818",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\footprints\M38\20200819",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\footprints\M38\20200820",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\footprints\M38\20200821",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\footprints\M38\20200824",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\footprints\M38\20200826",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\footprints\M38\20200827",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\footprints\M38\20200830",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\footprints\M38\20200902",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\footprints\M38\20200905",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\footprints\M38\20200908",
                     r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch3\batch_processing\cell_alignments\footprints\M38\20200911"]

# use pickle file or h5py file (default False)
is_pkl_file = False

# **********************************************************************************************************************

def extract_for_manualmatching(input_folderpaths, output_filepaths, isPickleFile=True):


    for file_idx, filepath in enumerate(input_folderpaths):
        print("Selected file: ", filepath + "\cnm_results.npy")

        # get tiff image
        print(filepath + "\mean_intensity_image.tif")
        mean_intensity_image = Image.open(filepath + "\mean_intensity_image.tif")

        # get relevant data from pickle file
        if isPickleFile:
            complete_filepath = filepath + "\pcf_results.pickle"
            print(complete_filepath)
            infile = open(complete_filepath, 'rb')
            data_complete = pickle.load(infile)
            print("loaded", complete_filepath)
            # print(data_complete.cnmf.estimates.A.shape)
            infile.close()
            new_data = {
                'dff_trace': data_complete.cnmf.estimates.F_dff,
                'spatial_masks': data_complete.cnmf.estimates.A,
                'template': data_complete.cnmf.estimates.Cn,
                'mean_intensity_template': np.array(mean_intensity_image)}
        else:
            print("to load: ", filepath + "\cnm_results.hdf5")
            complete_filepath = filepath + "\cnm_results.hdf5"
            with h5py.File(complete_filepath, "r") as f:
                print(f["estimates"]["A"]["shape"][()])
                spatial_mask_mtx = sparse.csc_matrix((f["estimates"]["A"]["data"],
                                                      f["estimates"]["A"]["indices"],
                                                      f["estimates"]["A"]["indptr"]),
                                                     shape=f["estimates"]["A"]["shape"][()])
                spatial_mask_mtx.todense()
                new_data = {
                    'dff_trace': f["estimates"]["F_dff"][()],
                    'spatial_masks': spatial_mask_mtx,
                    'template': f["estimates"]["Cn"][()],
                    'mean_intensity_template': np.array(mean_intensity_image)}
        np.save(output_filepaths[file_idx], new_data)


def main():
    extract_for_manualmatching(input_folderpaths, output_filepaths, isPickleFile=is_pkl_file)

if __name__ == "__main__":
    main()