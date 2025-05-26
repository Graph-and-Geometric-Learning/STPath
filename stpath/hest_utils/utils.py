import os
import pickle
import datetime
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]
NONE_MEAN = None
NONE_STD = None


def get_constants(norm='imagenet'):
    if norm == 'imagenet':
        return IMAGENET_MEAN, IMAGENET_STD
    elif norm == 'openai_clip':
        return OPENAI_MEAN, OPENAI_STD
    elif norm == 'none':
        return NONE_MEAN, NONE_STD
    else:
        raise ValueError(f"Invalid norm: {norm}")


def get_eval_transforms(mean, std, target_img_size = -1, center_crop = False):
    trsforms = []
    
    if target_img_size > 0:
        trsforms.append(transforms.Resize(target_img_size))
    if center_crop:
        assert target_img_size > 0, "target_img_size must be set if center_crop is True"
        trsforms.append(transforms.CenterCrop(target_img_size))
        
    
    trsforms.append(transforms.ToTensor())
    if mean is not None and std is not None:
        trsforms.append(transforms.Normalize(mean, std))
    trsforms = transforms.Compose(trsforms)

    return trsforms


def get_current_time():
    now = datetime.datetime.now()
    year = now.year % 100  # convert to 2-digit year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second
    return f"{year:02d}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}-{second:02d}"


def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()


def merge_dict(main_dict, new_dict, value_fn = None):
    """
    Merge new_dict into main_dict. If a key exists in both dicts, the values are appended. 
    Else, the key-value pair is added.
    Expects value to be an array or list - if not, it is converted to a list.
    If value_fn is not None, it is applied to each item in each value in new_dict before merging.
    Args:
        main_dict: main dict
        new_dict: new dict
        value_fn: function to apply to each item in each value in new_dict before merging
    """
    if value_fn is None:
        value_fn = lambda x: x
    for key, value in new_dict.items():
        if not isinstance(value, list):
            value = [value]
        value = [value_fn(v) for v in value]
        if key in main_dict:
            main_dict[key] = main_dict[key] + value
        else:
            main_dict[key] = value
    return main_dict


def get_path(path):
    
    def get_path_relative(file, path) -> str:
        curr_dir = os.path.dirname(os.path.abspath(file))
        return os.path.join(curr_dir, path)

    src = get_path_relative(__file__, '../../../../')
    if path.startswith('./'):
        new_path = os.path.join(src, path)
    else:
        new_path = path
    return new_path