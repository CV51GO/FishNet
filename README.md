## 介绍

本项目是论文《FishNet: a versatile backbone for image, region, and pixel level prediction》的Megengine实现。该论文的官方实现地址：https://github.com/kevin-ssy/FishNet


## 环境安装

依赖于CUDA10

```
conda create -n FishNet python=3.7
pip install -r requirements.txt
```

下载官方的权重：https://pan.baidu.com/s/11U3sRod1VfbDBRbmXph6KA
，将下载后的文件置于./official_FishNet路径下。

## 使用方法

安装完环境后，直接运行`python compare.py`。

`compare.py`文件对官方实现和Megengine实现的推理结果进行了对比。

运行`compare.py`时，会读取`./data`中存放的图片进行推理。`compare.py`中实现了Megengine框架和官方使用的Pytorch框架的推理，并判断两者推理结果的一致性。

## 模型加载示例

在model.py中，定义了```get_megengine_fishnet_model```方法，该方法能够利用hub加载模型。
```python
@hub.pretrained(
"https://studio.brainpp.com/api/v1/activities/3/missions/92/files/8b51a6a2-391f-49e2-a202-60e8a9dda7fe"
)
def get_megengine_fishnet_model():
    model_megengine = fishnet99()
    return model_megengine
```

在使用模型时，使用如下代码即可加载权重：
```python
from official_FishNet.net_factory import fishnet99 as torch_fishnet99
megengine_model = get_megengine_fishnet99(pretrained=True)
```
