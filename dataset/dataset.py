# dataset.py

from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch

from dataset.data_processing import load_data


def prepare_dataloader_with_split(config, val_split=0.1):
    dataset = AudioFacialDataset(config)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=AudioFacialDataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=AudioFacialDataset.collate_fn)

    return train_dataset, val_dataset, train_dataloader, val_dataloader

def prepare_dataloader(config):
    dataset = AudioFacialDataset(config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=AudioFacialDataset.collate_fn)
    return dataset, dataloader

class AudioFacialDataset(Dataset):
    def __init__(self, config):
        self.root_dir = config['root_dir']
        
        self.micro_batch_size = config['micro_batch_size']
        self.examples = []
        self.processed_folders = set()

        raw_examples = load_data(self.root_dir, self.processed_folders)
        
        self.examples = []
        for small_frames, large_frames in raw_examples:
            processed_examples = self.process_example(small_frames, large_frames)
            if processed_examples is not None:
                self.examples.extend(processed_examples)
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
        trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=0)
        return src_batch, trg_batch

    def process_example(self, small_frames, large_frames):
        num_frames_small = len(small_frames)
        num_frames_large = len(large_frames)
        max_frames = max(num_frames_small, num_frames_large)

        examples = []
        # Step by micro_batch_size instead of 1
        for start in range(0, max_frames - self.micro_batch_size + 1, self.micro_batch_size):
            end = start + self.micro_batch_size
            
            small_segment = np.zeros((self.micro_batch_size, small_frames.shape[1]))
            large_segment = np.zeros((self.micro_batch_size, large_frames.shape[1]))
            
            # Fill in available frames (if the video has fewer frames than expected, this handles it)
            small_segment[:min(self.micro_batch_size, num_frames_small - start)] = small_frames[start:end]
            large_segment[:min(self.micro_batch_size, num_frames_large - start)] = large_frames[start:end]
            
            examples.append((
                torch.tensor(small_segment, dtype=torch.float32),
                torch.tensor(large_segment, dtype=torch.float32)
            ))
        
        # Handle any leftover frames if max_frames is not a multiple of micro_batch_size
        if max_frames % self.micro_batch_size != 0:
            start = max_frames - self.micro_batch_size
            end = max_frames

            small_segment = np.zeros((self.micro_batch_size, small_frames.shape[1]))
            large_segment = np.zeros((self.micro_batch_size, large_frames.shape[1]))
            
            segment_small = small_frames[start:end]
            segment_large = large_frames[start:end]
            
            # Reflect to fill the remaining micro-batch if necessary
            reflection_small = np.flip(segment_small, axis=0)
            reflection_large = np.flip(segment_large, axis=0)
            
            small_segment[:len(segment_small)] = segment_small
            small_segment[len(segment_small):] = reflection_small[:self.micro_batch_size - len(segment_small)]
            
            large_segment[:len(segment_large)] = segment_large
            large_segment[len(segment_large):] = reflection_large[:self.micro_batch_size - len(segment_large)]
            
            examples.append((
                torch.tensor(small_segment, dtype=torch.float32),
                torch.tensor(large_segment, dtype=torch.float32)
            ))
        
        return examples
"""
#if you want to go line by line with a reduced frame size, you can use this instead - this only moves one row per grab of pairs within a frame

# dataset.py

from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch

from dataset.data_processing import load_data


def prepare_dataloader_with_split(config, val_split=0.1):
    dataset = AudioFacialDataset(config)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=AudioFacialDataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=AudioFacialDataset.collate_fn)

    return train_dataset, val_dataset, train_dataloader, val_dataloader

def prepare_dataloader(config):
    dataset = AudioFacialDataset(config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=AudioFacialDataset.collate_fn)
    return dataset, dataloader

class AudioFacialDataset(Dataset):
    def __init__(self, config):
        self.root_dir = config['root_dir']
        
        self.micro_batch_size = config['micro_batch_size']
        self.examples = []
        self.processed_folders = set()

        raw_examples = load_data(self.root_dir, self.processed_folders)
        
        self.examples = []
        for small_frames, large_frames in raw_examples:
            processed_examples = self.process_example(small_frames, large_frames)
            if processed_examples is not None:
                self.examples.extend(processed_examples)
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
        trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=0)
        return src_batch, trg_batch

    def process_example(self, small_frames, large_frames):

        num_frames_small = len(small_frames)
        num_frames_large = len(large_frames)

        max_frames = max(num_frames_small, num_frames_large)

        examples = []
        for start in range(0, max_frames - self.micro_batch_size + 1):
            end = start + self.micro_batch_size
            
            small_segment = np.zeros((self.micro_batch_size, small_frames.shape[1]))
            large_segment = np.zeros((self.micro_batch_size, large_frames.shape[1]))
            
            small_segment[:min(self.micro_batch_size, num_frames_small - start)] = small_frames[start:end]
            large_segment[:min(self.micro_batch_size, num_frames_large - start)] = large_frames[start:end]

            examples.append((torch.tensor(small_segment, dtype=torch.float32), torch.tensor(large_segment, dtype=torch.float32)))

        if max_frames % self.micro_batch_size != 0:
            start = max_frames - self.micro_batch_size
            end = max_frames

            small_segment = np.zeros((self.micro_batch_size, small_frames.shape[1]))
            large_segment = np.zeros((self.micro_batch_size, large_frames.shape[1]))
            
            segment_small = small_frames[start:end]
            segment_large = large_frames[start:end]

            reflection_small = np.flip(segment_small, axis=0)
            reflection_large = np.flip(segment_large, axis=0)

            small_segment[:len(segment_small)] = segment_small
            small_segment[len(segment_small):] = reflection_small[:self.micro_batch_size - len(segment_small)]

            large_segment[:len(segment_large)] = segment_large
            large_segment[len(segment_large):] = reflection_large[:self.micro_batch_size - len(segment_large)]

            examples.append((torch.tensor(small_segment, dtype=torch.float32), torch.tensor(large_segment, dtype=torch.float32)))
        
        return examples """
