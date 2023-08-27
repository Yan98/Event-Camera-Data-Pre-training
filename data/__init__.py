import importlib
import torch
import torch.utils.data
from os import path as osp

from utils import get_root_logger, scandir
from torch.utils.data import DataLoader

__all__ = ['create_dataset', 'create_dataloader']

# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(data_folder)
    if v.endswith('_dataset.py')
]
# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f'data.{file_name}')
    for file_name in dataset_filenames
]


def create_dataset(dataset_opt,logger_name):
    """Create dataset.
    Args:
        dataset_opt (dict): Configuration for dataset. It constains:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_type = dataset_opt['type']

    # dynamic instantiation
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')

    dataset = dataset_cls(dataset_opt)

    logger = get_root_logger(logger_name)
    logger.info(
        f'Dataset {dataset.__class__.__name__} - {dataset_opt["type"]} '
        'is created.')
    return dataset


def create_dataloader(dataset,
                      dataset_opt,
                      logger_name,
                      num_process,
                      ):
    
    phase = dataset_opt['phase']
    logger = get_root_logger(logger_name)
    
    collate_fn = None

    if phase == 'train':
        sampler = torch.utils.data.DistributedSampler(
            dataset, shuffle=True
        )
        TrainLoader = DataLoader(dataset, 
            batch_size=dataset_opt["batch_size"], num_workers=dataset_opt["num_workers"], drop_last=True, pin_memory=dataset_opt["pin_memory"], persistent_workers = dataset_opt["persistent_workers"], collate_fn = collate_fn, sampler = sampler)

        logger.info("Train loader is created")
        return TrainLoader
    elif phase== 'test' or phase== 'eval' :
        if len(dataset) % num_process == 0:
            sampler = torch.utils.data.DistributedSampler(
                dataset, shuffle=False,
            )
            logger.info("DDP eval")
        else:
           sampler = torch.utils.data.RandomSampler(dataset)
           logger.info("Sequential eval")
           
        TestLoader = DataLoader(dataset,
                batch_size=dataset_opt["batch_size"], num_workers=dataset_opt["num_workers"], drop_last=False, pin_memory= dataset_opt["pin_memory"],persistent_workers = dataset_opt["persistent_workers"], collate_fn = collate_fn, sampler = sampler)
        logger.info("Test loader is created")
        return TestLoader

    else:
        raise AttributeError('Mode not provided')
