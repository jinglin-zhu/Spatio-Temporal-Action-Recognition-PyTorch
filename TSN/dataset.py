import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class miniUCFTSNDataset(Dataset):
    def __init__(self, data_list_path, classes_path, modality='RGB', num_segments=4, mode='train'):
        """
        Initialize the miniUCF TSN Dual-Modality Dataset.
        
        Args:
            data_list_path (str): Absolute path to the split file (e.g., 'data/train.txt' 
                or 'data/validation.txt').
            classes_path (str): Absolute path to the class-to-index mapping file 
                (e.g., 'data/classes.txt').
            modality (str, optional): Input modality to load. Options: 'RGB' or 'Flow'. 
                Defaults to 'RGB'.
            num_segments (int, optional): Number of temporal segments K to divide each video into. 
                Defaults to 4.
            mode (str, optional): Phase of execution. Options: 'train' (training) 
                or 'val' (validation/testing). Defaults to 'train'.
        """
        self.modality = modality
        self.num_segments = num_segments
        self.mode = mode
        
        # Path configuration
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(script_dir)
        self.frames_root = os.path.join(self.project_root, 'data', 'mini_UCF_frames')
        self.flow_root = os.path.join(self.project_root, 'data', 'mini_UCF_flow')
        
        # 1. Load class mappings (ClassName -> ClassID)
        self.class_to_idx = {}
        with open(classes_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    idx = int(parts[0])
                    class_name = parts[1]
                    self.class_to_idx[class_name] = idx
                    
        # 2. Read video items and count total available frames
        self.video_items = []
        with open(data_list_path, 'r') as f:
            for line in f:
                video_rel_path = line.strip()
                if not video_rel_path:
                    continue
                
                class_name = video_rel_path.split('/')[0]
                label = self.class_to_idx[class_name]
                
                if self.modality == 'RGB':
                    video_dir = os.path.join(self.frames_root, video_rel_path)
                    if os.path.exists(video_dir):
                        num_frames = len([n for n in os.listdir(video_dir) if n.startswith('frame_')])
                        if num_frames > 0:
                            self.video_items.append((video_dir, num_frames, label))
                else:
                    # For Flow modality, check 'flow_x_' prefixes to count available flow frames
                    video_dir = os.path.join(self.flow_root, video_rel_path)
                    if os.path.exists(video_dir):
                        num_frames = len([n for n in os.listdir(video_dir) if n.startswith('flow_x_')])
                        if num_frames > 0:
                            self.video_items.append((video_dir, num_frames, label))
                            
        print(f"Successfully loaded TSN-{self.modality} ({self.mode}) dataset size: {len(self.video_items)}")

    def __len__(self):
        """
        Get the total number of samples in the dataset.
        
        Returns:
            int: Total number of video paths loaded in self.video_items.
        """
        return len(self.video_items)

    def _load_image(self, path, is_grayscale=False):
        """
        Load a single image file from disk and convert it to PIL format.
        
        Args:
            path (str): The absolute path of the target image file.
            is_grayscale (bool, optional): If True, convert the image to single-channel 'L' mode. 
                If False, convert the image to three-channel 'RGB' mode. Defaults to False.
        
        Returns:
            PIL.Image.Image: Loaded image object.
        """
        try:
            if is_grayscale:
                return Image.open(path).convert('L')
            return Image.open(path).convert('RGB')
        except Exception:
            # Fallback placeholder image in case of reading exceptions
            return Image.new('L' if is_grayscale else 'RGB', (256, 256))

    def _sample_indices(self, total_frames):
        """
        Segment the video and sample one index per segment (Sparse Temporal Sampling).
        
        Args:
            total_frames (int): Total available frames (RGB frames or optical flow stacks) in the folder.
        
        Returns:
            list of int: List of K integers (1-based frame indices) representing the sampled frames.
                For Flow mode, these act as starting indices for stacking 5 frames.
        """
        # For Flow modality, we stack 5 consecutive frames, 
        # so the starting index cannot exceed total_frames - 4
        valid_len = total_frames if self.modality == 'RGB' else (total_frames - 4)
        
        indices = []
        if valid_len >= self.num_segments:
            # Standard segmented sampling
            seg_len = valid_len / self.num_segments
            for i in range(self.num_segments):
                start = int(i * seg_len) + 1
                end = int((i + 1) * seg_len)
                if self.mode == 'train':
                    idx = random.randint(start, end)
                else:
                    idx = (start + end) // 2
                indices.append(idx)
        else:
            # Handle extremely short videos with loop-indexing
            for i in range(self.num_segments):
                if self.mode == 'train':
                    idx = random.randint(1, max(1, valid_len))
                else:
                    idx = (i % max(1, valid_len)) + 1
                indices.append(idx)
        return indices

    def _spatial_transform(self, images, is_training=True):
        """
        Apply identical spatial augmentations to all sampled images from the same video.
        
        Args:
            images (list of PIL.Image.Image): A list containing all the loaded PIL Image objects.
            is_training (bool, optional): If True, apply random spatial cropping (224x224) and 
                random horizontal flipping. If False, apply deterministic center cropping (224x224). 
                Defaults to True.
        
        Returns:
            list of torch.Tensor: List of normalized 3D tensors of shape [C, H, W], 
                where C=3 for RGB, and C=1 for Flow (grayscale).
        """
        transformed_images = []
        
        # 1. Determine consistent scaling (preserve aspect ratio, shorter side scaled to 256)
        w, h = images[0].size
        if w < h:
            new_w, new_h = 256, int(h * 256 / w)
        else:
            new_w, new_h = int(w * 256 / h), 256
            
        # 2. Pre-calculate consistent cropping and flipping parameters
        if is_training:
            # Randomly crop to 224x224
            # transforms.RandomCrop.get_params returns a random (i, j, h, w) bounding box
            dummy_img = Image.new('RGB', (new_w, new_h))
            crop_params = transforms.RandomCrop.get_params(dummy_img, output_size=(224, 224))
            flip = random.random() > 0.5
        else:
            # Center crop for validation/testing
            crop_params = (int((new_h - 224) / 2), int((new_w - 224) / 2), 224, 224)
            flip = False

        # 3. Apply transformation and normalization to each image
        for img in images:
            img_resized = TF.resize(img, [new_h, new_w])
            i, j, th, tw = crop_params
            img_cropped = TF.crop(img_resized, i, j, th, tw)
            if flip:
                img_cropped = TF.hflip(img_cropped)
                
            img_tensor = TF.to_tensor(img_cropped)
            
            # Normalize according to single or multi-channel dimensions
            # Optical flow mean is typically 0.5 (discretized to 0-255, center is 128)
            if img.mode == 'L':
                img_normalized = TF.normalize(img_tensor, mean=[0.5], std=[0.226])
            else:
                img_normalized = TF.normalize(img_tensor, 
                                              mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
            transformed_images.append(img_normalized)
            
        return transformed_images

    def __getitem__(self, idx):
        """
        Retrieve a sampled, augmented video tensor and its class label.
        
        Args:
            idx (int): Index of the target video item in self.video_items.
        
        Returns:
            tuple: A tuple containing:
                - stacked_tensor (torch.Tensor):
                    - If modality == 'RGB': 4D Tensor of shape [K, 3, 224, 224] 
                      (where K = num_segments).
                    - If modality == 'Flow': 4D Tensor of shape [K, 10, 224, 224] 
                      (10 channels: 5 consecutive x and y flows stacked).
                - label (int): The integer class index [0, 24] mapped from classes.txt.
        """
        video_dir, num_frames, label = self.video_items[idx]
        
        # 1. Sparse temporal sampling to retrieve start indices for 4 segments
        start_indices = self._sample_indices(num_frames)
        
        # 2. Load image resources according to the modality
        all_pil_images = []
        
        if self.modality == 'RGB':
            # RGB Modality: sample 1 frame from each of the 4 segments
            for f_idx in start_indices:
                frame_path = os.path.join(video_dir, f"frame_{f_idx:05d}.jpg")
                all_pil_images.append(self._load_image(frame_path, is_grayscale=False))
                
            # Apply spatially consistent augmentation
            tensors = self._spatial_transform(all_pil_images, is_training=(self.mode == 'train'))
            # Stack as: [K, 3, H, W] (i.e., [4, 3, 224, 224])
            stacked_tensor = torch.stack(tensors, dim=0)
            
        else:
            # Flow Modality: sample 5 consecutive x, y flow frames from each of the 4 segments
            # Each segment requires loading 5*2=10 single-channel images
            for start_idx in start_indices:
                for offset in range(5):
                    f_idx = start_idx + offset
                    flow_x_path = os.path.join(video_dir, f"flow_x_{f_idx:04d}.jpg")
                    flow_y_path = os.path.join(video_dir, f"flow_y_{f_idx:04d}.jpg")
                    
                    all_pil_images.append(self._load_image(flow_x_path, is_grayscale=True))
                    all_pil_images.append(self._load_image(flow_y_path, is_grayscale=True))
            
            # Spatially consistent augmentation across all 40 images (4 segments * 10 flow frames) to prevent temporal jitter
            tensors = self._spatial_transform(all_pil_images, is_training=(self.mode == 'train'))
            
            # Reshape to 4 segments with 10-channel format: [4, 10, 224, 224]
            segment_tensors = []
            for i in range(self.num_segments):
                # Extract 10 single-channel tensors for the i-th segment and concatenate along channel axis
                segment_flow = torch.cat(tensors[i*10 : (i+1)*10], dim=0) # [10, 224, 224]
                segment_tensors.append(segment_flow)
                
            # Stack as: [4, 10, 224, 224]
            stacked_tensor = torch.stack(segment_tensors, dim=0)
            
        return stacked_tensor, label