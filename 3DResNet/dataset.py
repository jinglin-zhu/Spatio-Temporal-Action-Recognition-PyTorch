import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class miniUCF3DDataset(Dataset):
    def __init__(self, data_list_path, classes_path, clip_len=16, mode='train', num_views=4):
        """
        Initialize the miniUCF 3D Action Recognition Dataset.

        Args:
            data_list_path (str): Path to train.txt or validation.txt split file.
            classes_path (str): Path to classes.txt class-to-index mapping file.
            clip_len (int, optional): Number of consecutive frames for 3D convolution input. 
                Defaults to 16.
            mode (str, optional): Phase of execution. Options: 'train' (training) 
                or 'val' (validation/testing). Defaults to 'train'.
            num_views (int, optional): Number of temporal views sampled during validation. 
                Defaults to 4.
        """
        self.clip_len = clip_len
        self.mode = mode
        self.num_views = num_views
        
        # dynamically retrieve the project root directory path to ensure robust path resolution
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(script_dir)
        self.frames_root = os.path.join(self.project_root, 'data', 'mini_UCF_frames')
        
        # 1. load class mappings (ClassName -> ClassID)
        self.class_to_idx = {}
        with open(classes_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    idx = int(parts[0])
                    class_name = parts[1]
                    self.class_to_idx[class_name] = idx
                    
        # 2. read video entries
        self.video_items = []
        with open(data_list_path, 'r') as f:
            for line in f:
                video_rel_path = line.strip() # in format "ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01"
                if not video_rel_path:
                    continue
                
                class_name = video_rel_path.split('/')[0]
                label = self.class_to_idx[class_name]
                
                # check the actual storage path of the frame images and calculate the frame count
                video_dir = os.path.join(self.frames_root, video_rel_path)
                if os.path.exists(video_dir):
                    # get the number of frame images in this video directory
                    num_frames = len([name for name in os.listdir(video_dir) if name.startswith('frame_')])
                    if num_frames > 0:
                        self.video_items.append((video_dir, num_frames, label))
                else:
                    print(f"Warning: Video frame folder not found.{video_dir}")
                    
        print(f"Successfully Loaded {self.mode} Sample Size: {len(self.video_items)}")

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Total number of valid video folders loaded in self.video_items.
        """
        return len(self.video_items)

    def _load_frame(self, video_dir, frame_idx):
        """
        Read a single JPEG frame from a video directory.

        Args:
            video_dir (str): The folder containing the frame images of a specific video.
            frame_idx (int): The 1-based index of the target frame image.

        Returns:
            PIL.Image.Image: The loaded frame converted to 'RGB' mode, 
                or a black image placeholder of size (256, 256) if reading fails.
        """
        frame_path = os.path.join(video_dir, f"frame_{frame_idx:05d}.jpg")
        try:
            return Image.open(frame_path).convert('RGB')
        except Exception as e:
            # if a read exception occurs, generate a solid black image as a background layer
            return Image.new('RGB', (256, 256))

    def _spatial_transform(self, images, is_training=True):
        """
        Applies data augmentation for spatial consistency to all frames within a clip.

        Args:
            images (list of PIL.Image.Image): A list of PIL Images representing a temporal clip.
            is_training (bool, optional): If True, apply random spatial cropping (224x224) 
                and random horizontal flipping. If False, apply deterministic center cropping (224x224). 
                Defaults to True.

        Returns:
            torch.Tensor: Normalized spatio-temporal tensor of shape [3, clip_len, 224, 224] 
                suitable for Conv3D layers.
        """
        transformed_images = []
        
        # 1. determine uniform scaling (proportional scaling such that the shortest side is 256)
        # the size of a PIL image is (width, height)
        w, h = images[0].size
        if w < h:
            new_w, new_h = 256, int(h * 256 / w)
        else:
            new_w, new_h = int(w * 256 / h), 256
            
        # 2. if in the training phase, randomly determine the spatial cropping parameters and flipping decisions
        if is_training:
            # randomly crop to 224x244
            # simulate a scaled virtual image to obtain the cropping frame
            dummy_img = Image.new('RGB', (new_w, new_h))
            crop_params = transforms.RandomCrop.get_params(dummy_img, output_size=(224, 224)) # returns a random (i, j, h, w) cropping frame
            flip = random.random() > 0.5
        else:
            # use center cropping during the validation phase
            crop_params = (int((new_h - 224) / 2), int((new_w - 224) / 2), 224, 224)
            flip = False

        # 3. apply the same operation to 16 images
        for img in images:
            # a. scale
            img_resized = TF.resize(img, [new_h, new_w])
            # b. crop
            i, j, th, tw = crop_params
            img_cropped = TF.crop(img_resized, i, j, th, tw)
            # c. flip horizontally
            if flip:
                img_cropped = TF.hflip(img_cropped)
            # d. normalize to tensor (ImageNet mean and standard deviation)
            img_tensor = TF.to_tensor(img_cropped)
            img_normalized = TF.normalize(img_tensor, 
                                          mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])
            transformed_images.append(img_normalized)
            
        # 4. stack along the temporal dimension
        clip_tensor = torch.stack(transformed_images, dim=0) # [clip_len, 3, 224, 224]
        clip_tensor = clip_tensor.permute(1, 0, 2, 3)        # [3, clip_len, 224, 224]
        return clip_tensor

    def __getitem__(self, idx):
        """
        Retrieve a sampled spatio-temporal tensor and its label.

        Args:
            idx (int): Index of the target video item in self.video_items.

        Returns:
            tuple: A tuple containing:
                - clip or stacked_views (torch.Tensor):
                    - If mode == 'train': Spatio-temporal tensor of shape [3, clip_len, 224, 224].
                    - If mode == 'val': Multi-view tensor of shape [num_views, 3, clip_len, 224, 224] 
                      where each view represents an evenly sampled temporal slice.
                - label (int): The integer class index [0, 24] mapped from classes.txt.
        """
        video_dir, num_frames, label = self.video_items[idx]
        
        if self.mode == 'train':
            # training mode: temporally randomly crop 1 spatio-temporal slice
            if num_frames >= self.clip_len:
                # randomly select the starting frame index
                start_idx = random.randint(1, num_frames - self.clip_len + 1)
                frame_indices = list(range(start_idx, start_idx + self.clip_len))
            else:
                # loop short videos
                frame_indices = [(i % num_frames) + 1 for i in range(self.clip_len)]
                
            # read and enhance image frames
            images = [self._load_frame(video_dir, f_idx) for f_idx in frame_indices]
            clip = self._spatial_transform(images, is_training=True) # [3, clip_len, 224, 224]
            return clip, label
            
        else:
            # validation mode: uniformly sample 4 temporal viewpoints (multi-view).
            # generate 4 starting frame points within the feasible range
            if num_frames >= self.clip_len:
                start_offsets = np.linspace(1, num_frames - self.clip_len + 1, num=self.num_views, dtype=int)
            else:
                # if the video is too short, directly copy the 4 padded slices
                start_offsets = [1] * self.num_views
                
            views_clips = []
            for start_idx in start_offsets:
                if num_frames >= self.clip_len:
                    frame_indices = list(range(start_idx, start_idx + self.clip_len))
                else:
                    frame_indices = [(i % num_frames) + 1 for i in range(self.clip_len)]
                
                # load the frame image corresponding to the specific viewpoint
                images = [self._load_frame(video_dir, f_idx) for f_idx in frame_indices]
                clip = self._spatial_transform(images, is_training=False) # [3, clip_len, 224, 224]
                views_clips.append(clip)
                
            # stack 4 perspectives together：[4, 3, clip_len, 224, 224]
            stacked_views = torch.stack(views_clips, dim=0)
            return stacked_views, label