import torch
from torch.utils.data import Dataset
import numpy as np
import random
import os
from .augmentation import get_augmentation, RandAug
from functools import partial

SENSOR_H = 480
SENSOR_W = 640
IMAGE_H = 224
IMAGE_W = 224
TIME_SCALE = 1000000

def load_event(event_path, cfg):
    # Returns time-shifted numpy array event from event_path
    event = np.load(event_path)
    if cfg.get('compressed', True):
        event = event['event_data']
        event = np.vstack([event['x'], event['y'], event['t'], event['p'].astype(np.uint8)]).T
    else:
        event = np.vstack([event['x_pos'], event['y_pos'], event['timestamp'], event['polarity'].astype(np.uint8)]).T

    event = event.astype(np.float32)

    # Account for int-type timestamp
    event[:, 2] /= TIME_SCALE

    # Account for zero polarity
    if event[:, 3].min() >= -0.5:
        event[:, 3][event[:, 3] <= 0.5] = -1

    return event


def slice_event(event, cfg):
    slice_method = cfg.get('slice_method', 'idx')
    if slice_method == 'idx':
        start = cfg.get('slice_start', None)
        end = cfg.get('slice_end', None)
        event = event[start:end]
    elif slice_method == 'time':
        start = cfg.get('slice_start', None)
        end = cfg.get('slice_end', None)
        event = event[(event[:, 2] > start) & (event[:, 2] < end)]
    elif slice_method == 'random':
        length = cfg.get('slice_length', None)
        slice_augment = cfg.get('slice_augment', False)

        #print(str(slice_augment), str(cfg["phase"] == 'train'))
        if slice_augment and cfg["phase"] == 'train':
            slice_augment_width = cfg.get('slice_augment_width', 0)
            length = random.randint(length - slice_augment_width, length + slice_augment_width)

        if len(event) > length:
            start = random.choice(range(len(event) - length + 1))
            event = event[start: start + length]

    return event


def reshape_event_with_sample(event, orig_h, orig_w, new_h, new_w):
    # Sample events
    sampling_ratio = (new_h * new_w) / (orig_h * orig_w)

    new_size = int(sampling_ratio * len(event))
    idx_arr = np.arange(len(event))

    sampled_arr = np.random.choice(idx_arr, size=new_size, replace=False)
    sampled_event = event[np.sort(sampled_arr)]

    # Rescale coordinates
    sampled_event[:, 0] *= (new_w / orig_w)
    sampled_event[:, 1] *= (new_h / orig_h)

    return sampled_event


def reshape_event_no_sample(event, orig_h, orig_w, new_h, new_w):
    event[:, 0] *= (new_w / orig_w)
    event[:, 1] *= (new_h / orig_h)

    return event


def reshape_event_unique(event, orig_h, orig_w, new_h, new_w):
    event[:, 0] *= (new_w / orig_w)
    event[:, 1] *= (new_h / orig_h)

    coords = event[:, :2].astype(np.int64)
    timestamp = (event[:, 2] * TIME_SCALE).astype(np.int64)
    min_time = timestamp[0]
    timestamp -= min_time

    key = coords[:, 0] + coords[:, 1] * new_w + timestamp * new_h * new_w
    _, unique_idx = np.unique(key, return_index=True)

    event = event[unique_idx]

    return event


def parse_event(event_path, cfg):
    event = load_event(event_path, cfg)
    
    event = torch.from_numpy(event)

    # Account for slicing
    slice_events = cfg.get('slice_events', False)

    if slice_events:
        event = slice_event(event, cfg)

    reshape = cfg.get('reshape', False)
    if reshape:
        reshape_method = cfg.get('reshape_method', 'no_sample')

        if reshape_method == 'no_sample':
            event = reshape_event_no_sample(event, SENSOR_H, SENSOR_W, IMAGE_H, IMAGE_W)
        else:
            raise SystemExit

    return event

def reshape_then_acc_count_pol(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 2 * H * W image

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]

    # Get pos, neg counts
    pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W, minlength=H * W).reshape(H, W)
    neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W, minlength=H * W).reshape(H, W)

    ##
    #pos_count = pos_count / (pos_count.max() + 1)
    #neg_count = neg_count / (neg_count.max() + 1)
    ##

    result = torch.stack([pos_count, neg_count], dim=2)

    result = result.permute(2, 0, 1)
    result = result.float()
    return result

def random_shift_events(event_tensor, max_shift=20, resolution=(224, 224)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=(2,))
    event_tensor[:, 0] += x_shift
    event_tensor[:, 1] += y_shift

    valid_events = (event_tensor[:, 0] >= 0) & (event_tensor[:, 0] < W) & (event_tensor[:, 1] >= 0) & (event_tensor[:, 1] < H)
    event_tensor = event_tensor[valid_events]

    return event_tensor


def random_flip_events_along_x(event_tensor, resolution=(224, 224), p=0.5):
    H, W = resolution

    if np.random.random() < p:
        event_tensor[:, 0] = W - 1 - event_tensor[:, 0]

    return event_tensor


def random_time_flip(event_tensor, resolution=(224, 224), p=0.5):
    if np.random.random() < p:
        event_tensor = torch.flip(event_tensor, [0])
        event_tensor[:, 2] = event_tensor[0, 2] - event_tensor[:, 2]
        event_tensor[:, 3] = - event_tensor[:, 3]  # Inversion in time means inversion in polarity
    return event_tensor


def add_correlated_events(event, xy_std = 1.5, ts_std = 0.001, add_noise=0):
    if event.size(0) < 1000:
        return event
    to_add = np.random.randint(min(100, event.size(0)-1),min(5000,event.size(0)))
    event_new = torch.cat((
        event[:,[0]] + torch.normal(0, xy_std,size = (event.size(0),1)),
        event[:,[1]] + torch.normal(0, xy_std,size = (event.size(0),1)),
        event[:,[2]] + torch.normal(0, ts_std,size = (event.size(0),1)),
        event[:,[3]]
        ),-1)
    
    idx = np.random.choice(np.arange(event_new.size(0)), size=to_add, replace=False)
    event_new = event_new[idx]
    event_new[:,[0]] = torch.clip(event_new[:,[0]],0,event[:,[0]].max())
    event_new[:,[1]] = torch.clip(event_new[:,[1]],0,event[:,[1]].max())
    
    event = torch.cat((event,event_new))
    return event[event[:,2].argsort(descending = False)]  

def base_augment(mode):
    assert mode in ['train', 'eval']

    if mode == 'train':
        def augment(event):
            event = random_time_flip(event, resolution=(IMAGE_H, IMAGE_W))
            event = random_shift_events(event)
            event = add_correlated_events(event)
            return event
        return augment

    elif mode == 'eval':
        return None

def get_loader_type(loader_type):
    # Choose loader ((N, 4) event tensor -> Network input)
    if loader_type is None or loader_type == 'reshape_then_acc_count_pol':
        loader = reshape_then_acc_count_pol
    else:
        raise SystemExit
    return loader

def remove_hot_pixels(x):
    mask = x[x > 0.1]
    mask = mask.mean()
    x = torch.clip(x,max=mask)
    return x 

class ImageNetDataset(Dataset):
    def __init__(self, cfg):
        super(ImageNetDataset, self).__init__()
        self.mode = cfg["phase"]
        root = cfg["root"]        
        self.file = [os.path.join(root, i.strip()) for i in open(cfg["file"], 'r').readlines()]
        self.label = sorted(os.listdir(cfg["label_map"]))
        assert len(self.label) == 1000
        self.cfg = cfg
        self.augment_type = cfg.get('augment_type', None)
        self.loader_type = cfg.get('loader_type', None)
        self.event_parser = self.augment_parser(parse_event)

        self.loader = get_loader_type(self.loader_type)
   
        self.img_augmentation = get_augmentation(cfg)
        
        if cfg.get("remove_hot_pixels", False):
            self.post_fn1 = remove_hot_pixels
            print("remove_hot_pixels ")
        else:
            self.post_fn1 = lambda x:x
           
        if cfg.get("rand_aug", False) and self.mode == 'train':
            self.post_fn2 = RandAug()
            print("rand_aug ")
        else:
            self.post_fn2 = lambda x:x    
          
            
    def augment_parser(self, parser):
        def new_parser(event_path):
            return parser(event_path, self.cfg)
        return new_parser

    def get_label(self, name):
        name = name.split(os.sep)[-2]
        label = self.label.index(name)
        return torch.LongTensor([label])
    
    def __getitem__(self, idx):
        event_path = self.file[idx]
        label = self.get_label(event_path)
        # Load and optionally reshape event from event_path
        event = self.event_parser(event_path)
        augment_mode = 'train' if self.mode == 'train' else 'eval'
        event = self.loader(event, augment=base_augment(augment_mode), neglect_polarity=self.cfg.get('neglect_polarity', False),
            global_time=self.cfg.get('global_time', True), strict=self.cfg.get('strict', False), use_image=self.cfg.get('use_image', True),
            denoise_sort=self.cfg.get('denoise_sort', False), denoise_image=self.cfg.get('denoise_image', False),
            filter_flash=self.cfg.get('filter_flash', False), filter_noise=self.cfg.get('filter_noise', False),
            quantize_sort=self.cfg.get('quantize_sort', None))


        assert event.size(0) == 2
        
        data = {
            "event": self.img_augmentation(event) if self.mode == 'train' else event, 
            "label": label,
            }
        data["event"] = self.post_fn1(data["event"])
        data["event"] = data["event"] / (data["event"].amax([1,2],True) + 1)
        data["event"] = self.post_fn2(data["event"])
        
        return data

    def __len__(self):
        return len(self.file)

class PretrainImageNetDataset(ImageNetDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.emb={}
        self.point_level_aug = cfg.get("point_level_aug", False)
        self.img_augmentation_view1, self.img_augmentation_view2 = get_augmentation(cfg, pop_resize = self.point_level_aug)
        self.loader = partial(self.loader,
                              augment=base_augment(self.mode), neglect_polarity=self.cfg.get('neglect_polarity', False),
                              global_time=self.cfg.get('global_time', True), strict=self.cfg.get('strict', False), use_image=self.cfg.get('use_image', True),
                              denoise_sort=self.cfg.get('denoise_sort', False), denoise_image=self.cfg.get('denoise_image', False),
                              filter_flash=self.cfg.get('filter_flash', False), filter_noise=self.cfg.get('filter_noise', False),
                              quantize_sort=self.cfg.get('quantize_sort', None)
                              )
        
        if self.point_level_aug:
            raise SystemExit
        self.jitter = lambda x:x
                
    def get_emb(self,name):
        folder = name.split(os.sep)[-2]
        name = name.split(os.sep)[-1][:-4]
        if self.cfg.get("save_emb", True):
            if folder not in self.emb:
                self.emb[folder] = torch.load(os.path.join(self.cfg["emb_path"],folder + ".pt")) 
            return self.emb[folder][name].float().squeeze()
        else:
            return torch.load(os.path.join(self.cfg["emb_path"],folder + ".pt"))[name].float().squeeze() 
    
    def get_events(self, event):
        if not self.point_level_aug:
            event = self.loader(event) 
            event = self.post_fn1(event)
            event = event / (event.amax([1,2],True) + 1)
            event = self.post_fn2(event)
            event1, event2 = event,event
        else:
            raise SystemExit
        return self.jitter(event1), self.jitter(event2)
            
    
    def __getitem__(self, idx):
        
        event_path = self.file[idx]
        label = self.get_label(event_path)
        
        if "emb_path" in self.cfg:
            emb = self.get_emb(event_path)
        else:
            emb = torch.ones(1)
        
        event = self.event_parser(event_path)
        event1, event2 = self.get_events(event)
        
        data = {
            "event1": self.img_augmentation_view1(event1) if self.mode == 'train' else event1, 
            "event2": self.img_augmentation_view2(event2) if self.mode == 'train' else event2, #event2,
            "emb": emb,
            "label": label, 
            }
        
        return data
    
