import os
import sys
import numpy as np
import torch

from highd_utils.process_utils import *

# Specify Processing options
options = {}

# free options
options['min_track'] = 1 # min number to load
options['max_track'] = 60 # max number to load
options['min_track_length'] = 200 # min track frame length
options['max_track_length'] = 350 # max track frame length (will truncate beginning of trajectory)
options['p_keep_straight'] = 0.0 # Probability to keep a straight car (0.0 = lane changes only, 1.0 = full dataset)
options['frame_rate'] = 5 # Keep every nth frame. Note recorded frame rate is 25 Hz
options['locations'] = [1,2,3,4,5,6]
options['features'] = ['x','y','dhw','thw']

# fixed options
options['randseed'] = 1
options['input_path'] = "./raw/highD-dataset-v1.0/data/"
options['track_files'] = "%02d_tracks.csv"
options['track_meta_files'] = "%02d_tracksMeta.csv"
options['rec_meta_files'] = "%02d_recordingMeta.csv"
options['output_dir'] = "./processed/"
options['output_file'] = "highd_processed_tracks%02d-%02d_fr%02d_loc%s_p%3.2f.pt" %(
    options['min_track'],options['max_track'], 
    options['frame_rate'], "".join([str(i) for i in options['locations']]), 
    options['p_keep_straight'])

# Read all data
all_tracks, all_static, all_rec = read_all_data(options)

# Clean and combine all data
combined = combine_data(all_tracks, options)

# Map data to torch tensor with correct shape
data_tensor = feature_dict_to_tensor(combined, options)

# Reduce frame rate
red_data_tensor = reduce_frame_rate(data_tensor, options)

# Save data
save_tensor(red_data_tensor, options)