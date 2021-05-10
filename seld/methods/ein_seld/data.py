from pathlib import Path
from timeit import default_timer as timer

import h5py
import numpy as np
import torch
from methods.utils.data_utilities import (_segment_index, load_dcase_format,
                                          to_metrics2020_format, add_real_life_noise)
from methods.ein_seld.data_augmentation.pitch_shift import apply_pitch_shift
from methods.ein_seld.data_augmentation.channel_rotation import *
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from utils.common import int16_samples_to_float32


class UserDataset(Dataset):
    """ User defined datset

    """
    def __init__(self, args, cfg, dataset, dataset_type='train', overlap=''):
        """
        Args:
            args: input args
            cfg: configurations
            dataset: dataset used
            dataset_type: 'train' | 'valid' | 'dev_test' | 'eval_test'
            overlap: '1' | '2'
        """
        super().__init__()
        self.dataset_type = dataset_type
        self.read_into_mem = args.read_into_mem
        self.sample_rate = cfg['data']['sample_rate']
        self.clip_length = dataset.clip_length
        self.label_resolution = dataset.label_resolution
        self.frame_length = int(self.clip_length / self.label_resolution)
        self.label_interp_ratio = int(self.label_resolution * self.sample_rate / cfg['data']['hop_length'])
        self.cfg = cfg
        self.single_file = self.cfg['training']['single_file']

        # Chunklen and hoplen and segmentation. Since all of the clips are 60s long, it only segments once here
        data = np.zeros((1, self.clip_length * self.sample_rate))
        if 'train' in self.dataset_type:
            chunklen = int(cfg['data']['train_chunklen_sec'] * self.sample_rate)     
            hoplen = int(cfg['data']['train_hoplen_sec'] * self.sample_rate)
            self.segmented_indexes, self.segmented_pad_width = _segment_index(data, chunklen, hoplen)
        elif self.dataset_type in ['valid', 'dev_test', 'eval_test']:
            chunklen = int(cfg['data']['test_chunklen_sec'] * self.sample_rate)
            hoplen = int(cfg['data']['test_hoplen_sec'] * self.sample_rate)
            self.segmented_indexes, self.segmented_pad_width = _segment_index(data, chunklen, hoplen, last_frame_always_paddding=True)
        self.num_segments = len(self.segmented_indexes)
        # to train with add noise get the files of the noise dataset - esc50
        if self.cfg['data_noise']['add_noise']:
            # Path for dataset
            hdf5_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset_noise_dir'])
            # the noise sampling rate is equal to the data sampling rate
            data_h5_dir = hdf5_dir.joinpath('noise_dataset').joinpath('{}fs'.format(cfg['data']['sample_rate']))
            self.noise_data_h5_dir_list = data_h5_dir.joinpath('esc50')
            data_noise_dirs= self.noise_data_h5_dir_list
            self.paths_noise_list = [path for path in data_noise_dirs.glob('*.h5')]
        # Data and meta path
        fold_str_idx = dataset.fold_str_index
        ov_str_idx = dataset.ov_str_index
        data_sr_folder_name = '{}fs'.format(self.sample_rate)
        main_data_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset']).joinpath('data').joinpath(data_sr_folder_name)
        dev_data_dir = main_data_dir.joinpath('dev').joinpath(cfg['data']['type'])
        eval_data_dir = main_data_dir.joinpath('eval').joinpath(cfg['data']['type'])
        main_meta_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset']).joinpath('meta')
        dev_meta_dir = main_meta_dir.joinpath('dev')
        eval_meta_dir = main_meta_dir.joinpath('eval')
        if self.dataset_type == 'train':
            data_dirs = [dev_data_dir]
            self.meta_dir = dev_meta_dir
            train_fold = [int(fold.strip()) for fold in str(cfg['training']['train_fold']).split(',')]
            ov_set = str(cfg['training']['overlap']) if not overlap else overlap
            self.paths_list = [path for data_dir in data_dirs for path in sorted(data_dir.glob('*.h5')) \
                if int(path.stem[fold_str_idx]) in train_fold and path.stem[ov_str_idx] in ov_set \
                    and not path.name.startswith('.')]
        elif self.dataset_type == 'train_spec_aug':
            data_dirs = [dev_data_dir]
            self.meta_dir = dev_meta_dir
            train_fold = [int(fold.strip()) for fold in str(cfg['training']['train_fold']).split(',')]
            ov_set = str(cfg['training']['overlap']) if not overlap else overlap
            self.paths_list = [path for data_dir in data_dirs for path in sorted(data_dir.glob('*.h5')) \
                               if int(path.stem[fold_str_idx]) in train_fold and path.stem[ov_str_idx] in ov_set \
                               and not path.name.startswith('.')]

        elif self.dataset_type == 'train_rotate_channel':
            data_dirs = [dev_data_dir]
            self.meta_dir = dev_meta_dir
            train_fold = [int(fold.strip()) for fold in str(cfg['training']['train_fold']).split(',')]
            ov_set = str(cfg['training']['overlap']) if not overlap else overlap
            self.paths_list = [path for data_dir in data_dirs for path in sorted(data_dir.glob('*.h5')) \
                               if int(path.stem[fold_str_idx]) in train_fold and path.stem[ov_str_idx] in ov_set \
                               and not path.name.startswith('.')]
        elif self.dataset_type == 'train_invert_position_aug':
            data_dirs = [dev_data_dir]
            self.meta_dir = dev_meta_dir
            train_fold = [int(fold.strip()) for fold in str(cfg['training']['train_fold']).split(',')]
            ov_set = str(cfg['training']['overlap']) if not overlap else overlap
            self.paths_list = [path for data_dir in data_dirs for path in sorted(data_dir.glob('*.h5')) \
                               if int(path.stem[fold_str_idx]) in train_fold and path.stem[ov_str_idx] in ov_set \
                               and not path.name.startswith('.')]
        elif self.dataset_type == 'valid':
            if cfg['training']['valid_fold'] != 'eval':
                data_dirs = [dev_data_dir]
                self.meta_dir = dev_meta_dir
                valid_fold = [int(fold.strip()) for fold in str(cfg['training']['valid_fold']).split(',')]
                ov_set = str(cfg['training']['overlap']) if not overlap else overlap
                self.paths_list = [path for data_dir in data_dirs for path in sorted(data_dir.glob('*.h5')) \
                    if int(path.stem[fold_str_idx]) in valid_fold and path.stem[ov_str_idx] in ov_set \
                        and not path.name.startswith('.')]
                ori_meta_dir = Path(cfg['dataset_dir']).joinpath('metadata_dev')
            else:
                data_dirs = [eval_data_dir]
                self.meta_dir = eval_meta_dir
                ov_set = str(cfg['training']['overlap']) if not overlap else overlap
                self.paths_list = [path for data_dir in data_dirs for path in sorted(data_dir.glob('*.h5')) \
                    if not path.name.startswith('.')]
                ori_meta_dir = Path(cfg['dataset_dir']).joinpath('metadata_eval')
            frame_begin_index = 0
            self.valid_gt_sed_metrics2019 = []
            self.valid_gt_doa_metrics2019 = []
            self.valid_gt_dcaseformat = {}
            for path in self.paths_list:
                ori_meta_path = ori_meta_dir.joinpath(path.stem + '.csv')
                output_dict, sed_metrics2019, doa_metrics2019 = \
                    load_dcase_format(ori_meta_path, frame_begin_index=frame_begin_index, 
                        frame_length=self.frame_length, num_classes=len(dataset.label_set))
                self.valid_gt_dcaseformat.update(output_dict)
                self.valid_gt_sed_metrics2019.append(sed_metrics2019)
                self.valid_gt_doa_metrics2019.append(doa_metrics2019)
                frame_begin_index += self.frame_length
            self.valid_gt_sed_metrics2019 = np.concatenate(self.valid_gt_sed_metrics2019, axis=0)
            self.valid_gt_doa_metrics2019 = np.concatenate(self.valid_gt_doa_metrics2019, axis=0)
            self.gt_metrics2020_dict = to_metrics2020_format(self.valid_gt_dcaseformat, 
                self.valid_gt_sed_metrics2019.shape[0], label_resolution=self.label_resolution)  
        elif self.dataset_type == 'dev_test':
            data_dirs = [dev_data_dir]
            self.meta_dir = dev_meta_dir
            dev_test_fold = [int(fold.strip()) for fold in str(cfg['inference']['test_fold']).split(',')]
            ov_set = str(cfg['inference']['overlap']) if not overlap else overlap
            self.paths_list = [path for data_dir in data_dirs for path in sorted(data_dir.glob('*.h5')) \
                if int(path.stem[fold_str_idx]) in dev_test_fold and path.stem[ov_str_idx] in ov_set \
                    and not path.name.startswith('.')]
        elif self.dataset_type == 'eval_test':
            data_dirs = [eval_data_dir]
            self.meta_dir = eval_meta_dir
            self.paths_list = [path for data_dir in data_dirs for path in sorted(data_dir.glob('*.h5')) \
                if not path.name.startswith('.')]
        if self.single_file:
            self.paths_list = [Path(str(path)) for path in self.paths_list]
        else:
            # list of files segmented - each file has 15 segments.
            self.paths_list = [Path(str(path) + '%' + str(n)) for path in self.paths_list for n in
                               range(self.num_segments)]

        # Read into memory
        if self.read_into_mem:
            load_begin_time = timer()
            print('Start to load dataset: {}, ov={}......\n'.format(self.dataset_type + ' set', ov_set))
            iterator = tqdm(self.paths_list, total=len(self.paths_list), unit='clips')
            self.dataset_list = []
            for path in iterator:
                fn, n_segment = path.stem, int(path.name.split('%')[1])
                data_path = Path(str(path).split('%')[0])
                index_begin = self.segmented_indexes[n_segment][0]
                index_end = self.segmented_indexes[n_segment][1]
                pad_width_before = self.segmented_pad_width[n_segment][0]
                pad_width_after = self.segmented_pad_width[n_segment][1]
                with h5py.File(data_path, 'r') as hf:
                    x = int16_samples_to_float32(hf['waveform'][:, index_begin: index_end])
                pad_width = ((0, 0), (pad_width_before, pad_width_after))                    
                x = np.pad(x, pad_width, mode='constant')
                if 'test' not in self.dataset_type:
                    ov = fn[-1]
                    index_begin_label = int(index_begin / (self.sample_rate * self.label_resolution))
                    index_end_label = int(index_end / (self.sample_rate * self.label_resolution))
                    #pad_width_before_label = int(pad_width_before / (self.sample_rate * self.label_resolution))
                    pad_width_after_label = int(pad_width_after / (self.sample_rate * self.label_resolution))
                    meta_path = self.meta_dir.joinpath(fn + '.h5')
                    with h5py.File(meta_path, 'r') as hf:
                        sed_label = hf['sed_label'][index_begin_label: index_end_label, ...]
                        doa_label = hf['doa_label'][index_begin_label: index_end_label, ...] # NOTE: this is Catesian coordinates
                    if pad_width_after_label != 0:
                        sed_label_new = np.zeros((pad_width_after_label, 2, 14))
                        doa_label_new = np.zeros((pad_width_after_label, 2, 3))
                        sed_label = np.concatenate((sed_label, sed_label_new), axis=0)
                        doa_label = np.concatenate((doa_label, doa_label_new), axis=0)

                    self.dataset_list.append({
                        'filename': fn,
                        'n_segment': n_segment,
                        'ov': ov,
                        'waveform': x,
                        'sed_label': sed_label,
                        'doa_label': doa_label
                    })
                else:
                    self.dataset_list.append({
                        'filename': fn,
                        'n_segment': n_segment,
                        'waveform': x
                    })
            iterator.close()
            print('Loading dataset time: {:.3f}\n'.format(timer()-load_begin_time))

    def __len__(self):
        """Get length of the dataset

        """
        return len(self.paths_list)

    def __getitem__(self, idx):
        """
        Read features from the dataset
        """
        if self.read_into_mem:
            data_dict = self.dataset_list[idx]
            fn = data_dict['filename']
            n_segment = data_dict['n_segment']
            x = data_dict['waveform']
            if 'test' not in self.dataset_type:
                ov = data_dict['ov']
                sed_label = data_dict['sed_label']
                doa_label = data_dict['doa_label']
        if not self.read_into_mem and not self.single_file:
            # this line returns a segment
            path = self.paths_list[idx]
            # segment the file into 15 segments
            fn, n_segment = path.stem, int(path.name.split('%')[1])
            data_path = Path(str(path).split('%')[0])
            index_begin = self.segmented_indexes[n_segment][0]
            index_end = self.segmented_indexes[n_segment][1]
            pad_width_before = self.segmented_pad_width[n_segment][0]
            pad_width_after = self.segmented_pad_width[n_segment][1]
            if self.dataset_type == 'train_pitchshift':
                with h5py.File(data_path, 'r') as hf:
                    x = apply_pitch_shift(hf['waveform'][:, index_begin: index_end], self.sample_rate)
            #elif self.dataset_type == 'train_rotate_channel':
            #    with h5py.File(data_path, 'r') as hf:
            #        file, pattern = apply_data_channel_rotation('foa',int16_samples_to_float32(hf['waveform'][:]))
            #        x = file[:, index_begin: index_end]
            else:
                with h5py.File(data_path, 'r') as hf:
                    x = int16_samples_to_float32(hf['waveform'][:, index_begin: index_end])
            data_info = [index_begin, index_end, data_path]
            if self.cfg['data_noise']['add_noise']:
                # sample a random file from the noise dataset and add it to x
                noise_wave_path = random.choice(self.paths_noise_list)
                data_info.append(noise_wave_path)
                if not self.cfg['data_noise']['noisy_training'] and self.dataset_type == 'valid':
                    with h5py.File(noise_wave_path, 'r') as hf:
                        # [()] is the new way of getting the scalar value instead of .value
                        noise_wave = int16_samples_to_float32(hf['waveform'][()])
                    x = add_real_life_noise(x,noise_wave,data_info,self.cfg['data_noise']['SNR'], plot=False)
                if self.cfg['data_noise']['noisy_training']:
                    with h5py.File(noise_wave_path, 'r') as hf:
                        # [()] is the new way of getting the scalar value instead of .value
                        noise_wave = int16_samples_to_float32(hf['waveform'][()])
                    x = add_real_life_noise(x,noise_wave,data_info,self.cfg['data_noise']['SNR'], plot=False)
            pad_width = ((0, 0), (pad_width_before, pad_width_after))
            x = np.pad(x, pad_width, mode='constant')
            if 'test' not in self.dataset_type:
                ov = fn[-1]
                index_begin_label = int(index_begin / (self.sample_rate * self.label_resolution))
                index_end_label = int(index_end / (self.sample_rate * self.label_resolution))
                #pad_width_before_label = int(pad_width_before / (self.sample_rate * self.label_resolution))
                pad_width_after_label = int(pad_width_after / (self.sample_rate * self.label_resolution))
                meta_path = self.meta_dir.joinpath(fn + '.h5')
                with h5py.File(meta_path, 'r') as hf:
                    sed_label = hf['sed_label'][index_begin_label: index_end_label, ...]
                    doa_label = hf['doa_label'][index_begin_label: index_end_label, ...] # NOTE: this is Catesian coordinates
                if pad_width_after_label != 0:
                    sed_label_new = np.zeros((pad_width_after_label, 2, 14))
                    doa_label_new = np.zeros((pad_width_after_label, 2, 3))
                    sed_label = np.concatenate((sed_label, sed_label_new), axis=0)
                    doa_label = np.concatenate((doa_label, doa_label_new), axis=0)
                # invert data augmentation
                '''
                if self.cfg['training']['invert_position_aug']:
                    if np.random.random() > 0.5:
                        x = np.flip(x, axis=1)  # invert waveform time axis
                        sed_label = np.flip(sed_label, axis=0)  # invert sed label time axis
                        doa_label = np.flip(doa_label, axis=0)  # invert doa label time axis
                        doa_label = 0.0 - doa_label  # also invert sound source position
                 '''
        if self.single_file:
            sample = []
            # this line returns a segment
            path = self.paths_list[idx]
            # segment the path
            #segmeneted_file = [Path(str(path) + '%' + str(n)) for n in range(self.num_segments)]
            #fn, n_segment = path.stem, int(path.name.split('%')[1])
            fn  = path.stem 
            data_path = Path(str(path).split('%')[0])
            for i , n_segment in enumerate(self.segmented_indexes):
                index_begin = n_segment[0]
                index_end = n_segment[1]

                pad_width_before = self.segmented_pad_width[i][0]
                pad_width_after = self.segmented_pad_width[i][1]

                with h5py.File(data_path, 'r') as hf:
                    x = int16_samples_to_float32(hf['waveform'][:, index_begin: index_end])
                pad_width = ((0, 0), (pad_width_before, pad_width_after))
                x = np.pad(x, pad_width, mode='constant')
                if 'test' not in self.dataset_type:
                    ov = fn[-1]
                    index_begin_label = int(index_begin / (self.sample_rate * self.label_resolution))
                    index_end_label = int(index_end / (self.sample_rate * self.label_resolution))
                    #pad_width_before_label = int(pad_width_before / (self.sample_rate * self.label_resolution))
                    pad_width_after_label = int(pad_width_after / (self.sample_rate * self.label_resolution))
                    meta_path = self.meta_dir.joinpath(fn + '.h5')
                    with h5py.File(meta_path, 'r') as hf:
                        sed_label = hf['sed_label'][index_begin_label: index_end_label, ...]
                        doa_label = hf['doa_label'][index_begin_label: index_end_label, ...] # NOTE: this is Catesian coordinates
                    if pad_width_after_label != 0:
                        sed_label_new = np.zeros((pad_width_after_label, 2, 14))
                        doa_label_new = np.zeros((pad_width_after_label, 2, 3))
                        sed_label = np.concatenate((sed_label, sed_label_new), axis=0)
                        doa_label = np.concatenate((doa_label, doa_label_new), axis=0)

                if 'test' not in self.dataset_type:
                    single_sample = {
                        'filename': fn,
                        'data_type': self.dataset_type,
                        'n_segment': n_segment,
                        'ov': ov,
                        'waveform': x,
                        'sed_label': sed_label,
                        'doa_label': doa_label

                    }
                    sample.append(single_sample)
                else:
                    single_sample = {
                        'filename': fn,
                        'n_segment': n_segment,
                        'waveform': x,
                        'data_type': self.dataset_type
                    }
                    sample.append(single_sample)
            return sample

        if 'test' not in self.dataset_type:
            sample = {
                'filename': fn,
                'data_type': self.dataset_type,
                'n_segment': n_segment,
                'ov': ov,
                'waveform': x,
                'sed_label': sed_label,
                'doa_label': doa_label

            }
        else:
            sample = {
                'filename': fn,
                'n_segment': n_segment,
                'waveform': x,
                'data_type': self.dataset_type
            }
        return sample


class UserBatchSampler(Sampler):
    """User defined batch sampler. Only for train set.

    """
    def __init__(self, clip_num, batch_size, seed=2020):
        self.clip_num = clip_num
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)

        self.indexes = np.arange(self.clip_num)
        self.random_state.shuffle(self.indexes)
        self.pointer = 0
    
    def get_state(self):
        sampler_state = {
            'random': self.random_state.get_state(),
            'indexes': self.indexes,
            'pointer': self.pointer
        }
        return sampler_state

    def set_state(self, sampler_state):
        self.random_state.set_state(sampler_state['random'])
        self.indexes = sampler_state['indexes']
        self.pointer = sampler_state['pointer']
    
    def __iter__(self):
        """
        Return: 
            batch_indexes (int): indexes of batch
        """
        while True:
            if self.pointer >= self.clip_num:
                self.pointer = 0
                self.random_state.shuffle(self.indexes)
            
            batch_indexes = self.indexes[self.pointer: self.pointer + self.batch_size]
            self.pointer += self.batch_size
            yield batch_indexes

    def __len__(self):
        return (self.clip_num + self.batch_size - 1) // self.batch_size


class PinMemCustomBatch:
    def __init__(self, batch_dict, single_file):
        batch_fn = []
        batch_n_segment = []
        batch_ov = []
        batch_x = []
        batch_sed_label = []
        batch_doa_label = []
        batch_data_type = []

        if single_file:
            batch_dict = batch_dict[0]
            for n in range(len(batch_dict)):
                batch_fn.append(batch_dict[n]['filename'])
                batch_data_type.append(batch_dict[n]['data_type'])
                batch_n_segment.append(batch_dict[n]['n_segment'])
                batch_ov.append(batch_dict[n]['ov'])
                batch_x.append(batch_dict[n]['waveform'])
                batch_sed_label.append(batch_dict[n]['sed_label'])
                batch_doa_label.append(batch_dict[n]['doa_label'])
        else:
            for n in range(len(batch_dict)):
                batch_fn.append(batch_dict[n]['filename'])
                batch_data_type.append(batch_dict[n]['data_type'])
                batch_n_segment.append(batch_dict[n]['n_segment'])
                batch_ov.append(batch_dict[n]['ov'])
                batch_x.append(batch_dict[n]['waveform'])
                batch_sed_label.append(batch_dict[n]['sed_label'])
                batch_doa_label.append(batch_dict[n]['doa_label'])

        self.batch_out_dict = {
            'filename': batch_fn,
            'data_type': batch_data_type,
            'n_segment': batch_n_segment,
            'ov': batch_ov,
            'waveform': torch.tensor(batch_x, dtype=torch.float32),
            'sed_label': torch.tensor(batch_sed_label, dtype=torch.float32),
            'doa_label': torch.tensor(batch_doa_label, dtype=torch.float32),
        }

    def pin_memory(self):
        self.batch_out_dict['waveform'] = self.batch_out_dict['waveform'].pin_memory()
        self.batch_out_dict['sed_label'] = self.batch_out_dict['sed_label'].pin_memory()
        self.batch_out_dict['doa_label'] = self.batch_out_dict['doa_label'].pin_memory()
        return self.batch_out_dict


def collate_fn(batch_dict, single_file=False):
    """
    Merges a list of samples to form a mini-batch
    Pin memory for customized dataset
    """
    return PinMemCustomBatch(batch_dict, single_file)


class PinMemCustomBatchTest:
    def __init__(self, batch_dict):
        batch_fn = []
        batch_n_segment = []
        batch_x = []
        
        for n in range(len(batch_dict)):
            batch_fn.append(batch_dict[n]['filename'])
            batch_n_segment.append(batch_dict[n]['n_segment'])
            batch_x.append(batch_dict[n]['waveform'])

        self.batch_out_dict = {
            'filename': batch_fn,
            'n_segment': batch_n_segment,
            'waveform': torch.tensor(batch_x, dtype=torch.float32),
            'data_type': ['valid']
        }

    def pin_memory(self):
        self.batch_out_dict['waveform'] = self.batch_out_dict['waveform'].pin_memory()
        return self.batch_out_dict


def collate_fn_test(batch_dict):
    """
    Merges a list of samples to form a mini-batch
    Pin memory for customized dataset
    """
    return PinMemCustomBatchTest(batch_dict)
