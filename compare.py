import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from official_FishNet.net_factory import fishnet99 as torch_fishnet99

import megengine
import megengine.functional as F
from model import get_megengine_fishnet_model as get_megengine_fishnet99
megengine.set_log_level('ERROR', update_existing=True)
img_root = './data'
img_name_list = os.listdir(img_root)

def resize(img, size=256, interpolation=Image.BILINEAR):
    w, h = img.size
    short, long = (w, h) if w <= h else (h, w)
    if short == size:
        return img
    new_short, new_long = size, int(size * long / short)
    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    return img.resize((new_w, new_h), interpolation)

def center_crop(img, output_size=(224, 224)):
    image_width, image_height = list(img.size)
    crop_height, crop_width = output_size

    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
def ToTensor(pic):
    mode_to_nptype = {'I': np.int32, 'I;16': np.int16, 'F': np.float32}
    img = megengine.tensor(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.transpose((2, 0, 1))
    return  img.astype(np.float32)/255


def normalize(tensor, mean, std):
    dtype = tensor.dtype
    mean = megengine.tensor(mean, dtype=dtype)
    std = megengine.tensor(std, dtype=dtype)
    mean = mean.reshape(-1, 1, 1)
    std = std.reshape(-1, 1, 1)
    tensor = (tensor - mean) / std
    return tensor

# img_path = './data/640px-Cat_close-up_2004_b.jpg'

# 初始化Pytorch模型
torch_model = torch_fishnet99()
torch_model = torch.nn.DataParallel(torch_model)
torch_model.load_state_dict(torch.load('./official_FishNet/fishnet99_ckpt.tar')['state_dict'])
torch_model.eval()

# 初始化Megengine模型
megengine_model = get_megengine_fishnet99(pretrained=True)
megengine_model.eval()

for img_name in img_name_list:
    print(f'inference {img_name}')

    # 1.读取图片
    img = Image.open(Path(img_root)/img_name)

    resized_img = resize(img)
    crop_img = center_crop(resized_img)
    tensor_img = ToTensor(crop_img)

    mean_list = [0.485, 0.456, 0.406]
    std_list = [0.229, 0.224, 0.225]
    norm_img = normalize(tensor_img, mean_list, std_list)
        
    # 2.使用官方模型推理 
    with torch.no_grad():
        torch_output = torch_model(torch.from_numpy(norm_img.numpy()).unsqueeze(0))
    torch_score, torch_class_id = torch_output.topk(5, 1, True, True)

    # 3.使用Megengine模型推理
    megengine_output = megengine_model(F.expand_dims(norm_img, 0))
    megengine_score, megengine_class_id = F.topk(megengine_output, 5, True)

    # 4.比较官方模型和Megengine模型的输出
    np.testing.assert_allclose(megengine_score.numpy(), torch_score.cpu().numpy(), rtol=1e-3)
    np.testing.assert_allclose(torch_class_id.cpu().numpy(), megengine_class_id.numpy(), rtol=1e-3)
    print('pass')

