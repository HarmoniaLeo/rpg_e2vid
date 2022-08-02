import numpy as np
from src.io import npy_events_tools
from src.io import psee_loader
import tqdm
import os
from numpy.lib import recfunctions as rfn
import torch
import argparse
import torch.nn
from utils.loading_utils import load_model
from options.inference_options import set_inference_options
from image_reconstructor import ImageReconstructor
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='visualize one or several event files along with their boxes')
    parser.add_argument('-raw_dir', type=str)
    parser.add_argument('-label_dir', type=str)
    parser.add_argument('-target_dir', type=str)
    parser.add_argument('-dataset', type=str, default="gen4")
    parser.add_argument('-c', '--path_to_model', required=True, type=str,
                        help='path to model weights')
    parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
    parser.set_defaults(fixed_duration=False)
    parser.add_argument('-N', '--window_size', default=None, type=int,
                        help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
    parser.add_argument('-T', '--window_duration', default=33.33, type=float,
                        help="Duration of each event window, in milliseconds. Ignored if --fixed_duration=False")
    parser.add_argument('--num_events_per_pixel', default=0.35, type=float,
                        help='in case N (window size) is not specified, it will be \
                              automatically computed as N = width * height * num_events_per_pixel')
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=False)

    set_inference_options(parser)

    args = parser.parse_args()
    raw_dir = args.raw_dir
    label_dir = args.label_dir
    target_dir = args.target_dir
    dataset = args.dataset

    args.color = False

    model = load_model(args.path_to_model)
    device = torch.device('cuda:0')

    model = model.to(device)
    model.eval()

    if dataset == "gen4":
        # min_event_count = 800000
        shape = [720,1280]
        target_shape = [512, 640]
    elif dataset == "kitti":
        # min_event_count = 800000
        shape = [375,1242]
        target_shape = [192, 640]
    else:
        # min_event_count = 200000
        shape = [240,304]
        target_shape = [256, 320]
    events_window = 500000

    reconstructor = ImageReconstructor(model, shape[0], shape[1], model.num_bins, args)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for mode in ["train","val","test"]:
        file_dir = os.path.join(raw_dir, mode)
        root = file_dir
        label_root = os.path.join(label_dir, mode)
        target_root = os.path.join(target_dir, mode)
        if not os.path.exists(target_root):
            os.makedirs(target_root)
        try:
            files = os.listdir(file_dir)
        except Exception:
            continue
        # Remove duplicates (.npy and .dat)
        # files = files[int(2*len(files)/3):]
        #files = files[int(len(files)/3):]
        files = [time_seq_name[:-7] for time_seq_name in files
                        if time_seq_name[-3:] == 'dat']

        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)

        for i_file, file_name in enumerate(files):
            if not file_name == "17-04-13_15-05-43_3599500000_3659500000":
                continue
            # if not file_name == "moorea_2019-06-26_test_02_000_976500000_1036500000":
            #     continue
            event_file = os.path.join(root, file_name + '_td.dat')
            bbox_file = os.path.join(label_root, file_name + '_bbox.npy')
            #h5 = h5py.File(volume_save_path, "w")
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size, dtype = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()

            unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

            f_event = psee_loader.PSEELoader(event_file)

            #min_event_count = f_event.event_count()
            count_upper_bound = -100000000
            memory = None

            for bbox_count,unique_time in enumerate(unique_ts):
                end_time = int(unique_time)
                end_count = f_event.seek_time(end_time)
                if end_count is None:
                    continue
                start_count = end_count - events_window
                if start_count < 0:
                    start_count = 0
                if start_count <= count_upper_bound:
                    start_count = count_upper_bound
                
                dat_event = f_event
                dat_event.seek_event(start_count)

                events = dat_event.load_n_events(int(end_count - start_count))
                del dat_event
                events = rfn.structured_to_unstructured(events)[:, [0, 1, 2, 3]].astype(float)

                if not memory is None:
                    events = np.concatenate([memory, events])
                
                events = events[-events_window:]
                memory = events

                count_upper_bound = end_count

                last_timestamp = events[-1, 0]

                events_ = events.copy()
                events_[:,0] = events_[:,0] / 1000000

                event_tensor = events_to_voxel_grid_pytorch(events_,
                                                            num_bins=model.num_bins,
                                                            width=shape[1],
                                                            height=shape[0],
                                                            device=device)
                
                volume = reconstructor.update_reconstruction(event_tensor, 0, last_timestamp)

                volume = torch.nn.functional.interpolate(torch.from_numpy(volume[None,None,:,:]), size = target_shape, mode='nearest')
                save_dir = os.path.join(target_dir,"e2vid")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_dir = os.path.join(save_dir, mode)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                ecd = volume.cpu().numpy().copy()[0]
                
                ecd.astype(np.uint8).tofile(os.path.join(save_dir,file_name+"_"+str(unique_time)+".npy"))
                            
                torch.cuda.empty_cache()
            #h5.close()
            pbar.update(1)
        pbar.close()
        # if mode == "test":
        #     np.save(os.path.join(root, 'total_volume_time.npy'),np.array(total_volume_time))
        #     np.save(os.path.join(root, 'total_taf_time.npy'),np.array(total_taf_time))
        #h5.close()