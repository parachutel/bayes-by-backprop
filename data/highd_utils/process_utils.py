import os
import sys
import numpy as np
import torch
import tqdm

from highd_utils.read_csv import *

def read_all_data(options):
    """
    This method reads in the highD dataset given options
    :param options: the options for the preprocessing.
    :return: dicts for each recording of track info, static metadata, and recording metadata.
    """
    all_tracks, all_static, all_rec = {}, {}, {}
    np.random.seed(options['randseed']) #seed rng
    
    for track in range(options['min_track'],options['max_track']+1):
        print("Loading track %02d in %02d : %02d." %(track, options['min_track'], options['max_track']))

        # Make pathnames for current track
        if not os.path.exists(options['input_path']):
            print("Invalid input path. Make sure dataset is put in correct place")
            sys.exit(1)
        track_path = os.path.join(options['input_path'], (options['track_files'] % (track)))
        track_meta_path = os.path.join(options['input_path'], (options['track_meta_files'] % (track)))
        rec_meta_path = os.path.join(options['input_path'], (options['rec_meta_files'] % (track)))

        # Read the recording meta info
        try:
            meta_dictionary = read_meta_info(rec_meta_path)
        except:
            print("The video meta file is either missing or contains incorrect characters.")
            sys.exit(1)

        # Move on if not at desired location
        if meta_dictionary['locationId'] not in options['locations']:
            print("Track %02d not in locations." %(track))
            continue

        # Read the track meta info
        try:
            tracks = read_track_csv(track_path)
        except:
            print("The track file is either missing or contains incorrect characters.")
            sys.exit(1)

        # Read the track static meta info
        try:
            static_info = read_static_info(track_meta_path)
        except:
            print("The static info file is either missing or contains incorrect characters.")
            sys.exit(1)

        # Remove tracks that are too short or (probabilistically) have no lane change
        rm_keys = []
        for car in range(1,meta_dictionary['numVehicles']+1):
            if (static_info[car]['numLaneChanges'] == 0 and np.random.rand()>options['p_keep_straight']):
                rm_keys.append(car)
                continue
            if (static_info[car]['numFrames'] < options['min_track_length']):
                rm_keys.append(car)

        print("Removing %d out of %d frames." %(len(rm_keys),meta_dictionary['numVehicles']+1) )
        for car in reversed(rm_keys):
            del static_info[car] # dict keyed by car
            del tracks[car-1] # list indexed by car-1


        # Store in master dicts
        all_tracks[track] = tracks
        all_static[track] = static_info
        all_rec[track] = meta_dictionary
        
    return all_tracks, all_static, all_rec

def clean_feature(raw, options, feature):
    """
    This method cleans a single feature for a single car track.
    :param raw: the raw input feature.
    :param options: the options for the preprocessing.
    :param feature: the feature name being cleaned.
    :return: a 1d numpy array of the cleaned feature.
    """
    n = len(raw)
    data = np.array(raw)
    
    # if data too short, extend appropriately
    if n < options['max_track_length']:
        ext = options['max_track_length'] - n
        
        # features to store last value
        if feature in ['y','dhw','thw']:
            data = np.append(data,[data[-1]] * ext)
        
        # features to extrapolate
        if feature == 'x':
            vel = (data[-1] - data[-1-5])/5
            extrap = data[-1] + vel*np.arange(1,ext+1)
            data = np.concatenate((data,extrap))
    
    # truncate if over max track length
    elif n > options['max_track_length']:
        data = data[-options['max_track_length']:]
    
    # center x and y values
    if feature in ['x','y']:
        data = data - data[0]
    
    # flip negative x values
    if feature == 'x' and data[-1] < 0:
        data = -data
    
    # replace zeros in dhw and thw
    if feature == 'dhw':
        data[data==0] = 500 # 500m to collision if no car
    if feature == 'thw':
        data[data==0] = 15 # 15s to collision if no car
        
    # scale data
    if feature == 'x':
        data = data/700
    elif feature == 'y':
        data = data/8
    elif feature == 'dhw':
        data = data/500
    elif feature == 'thw':
        data = data/15
        
    return data

def combine_data(all_tracks, options):
    """
    This method extracts, cleans, and combines features over all saved recordings and tracks.
    :param all_tracks: a dictionary of recordings, each with a reduced list of cars.
    :param options: the options for the preprocessing.
    :return: a dict mapping feature name to a (cars, frames) cleaned numpy array.
    """
    # initialize data dictionary and data lists
    data = {}
    for feature in options['features']:
        data[feature] = np.zeros((0,options['max_track_length']))
    
    # iterate over all tracks and recordings, add data to numpy array
    with tqdm.tqdm(total=len(all_tracks.keys())) as pbar:
        for rec in all_tracks.keys():
            for car in all_tracks[rec]:
                for feature in options['features']:
                    
                    # clean feature (fix length, fix zeros, subtract initial points)
                    cleaned = clean_feature(car[feature], options, feature)
                    
                    # add to data vector
                    data[feature] = np.vstack((data[feature], cleaned))
            pbar.update(1)
    return data

def feature_dict_to_tensor(feature_dict, options):
    """
    This method takes a feature dictionary and maps it to a torch tensor of correct shape.
    :param feature_dict: a dictionary mapping feature name to a (# cars, # frames) numpy array.
    :param options: the options for the preprocessing.
    :return: a torch tensor of shape (# frames, # cars, # features)
    """
    
    # turn feature dict into tensor of shape (# features, # cars, #frames)
    data_tensor = torch.tensor([feature_dict[f] for f in options['features']])
    
    # permute to correct axes
    return data_tensor.permute(2,1,0)

def reduce_frame_rate(data_tensor, options):
    """
    This method takes a data tensor and reduces the frame rate in the first dimension based on specified options.
    :param data_tensor: a torch tensor of shape (# frames, # cars, # features).
    :param options: the options for the preprocessing.
    :return: a torch tensor of shape (# reduced frames, # cars, # features)
    """
    # reduce tensor based on the desired frame rate
    # get reduced frame indices
    reduced_frames = np.arange(0,data_tensor.shape[0],options['frame_rate'])
    return data_tensor[reduced_frames,:,:]

def save_tensor(data_tensor, options):
    """
    This method saves the processed torch tensor.
    :param data_tensor: the cleaned torch data tensor.
    :param options: the options for the preprocessing.
    """
    
    if not os.path.exists(options['output_dir']):
        os.mkdir(options['output_dir'])    
    torch.save(data_tensor, os.path.join(options['output_dir'], options['output_file']))