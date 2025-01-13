# WeCromCL: Weakly Supervised Cross-Modality Contrastive Learning for Transcription-only Supervised Text Spotting
![The frame work of WeCromCL](docs/framework.png)
This is the official implementation of [WeCromCL(ECCV 2024)](https://arxiv.org/pdf/2407.19507). You can now try to train or inference the model of first stage as described in our paper.
## To do
- Update the code and checkpoints in Stage2.
- Updata the ctw dataset and related code.

## Environment
We use [Anaconda](https://www.anaconda.com/) to create the environments based on cuda version 11.1.
```
conda create -n wecromcl python=3.8 -y
conda activate wecromcl
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
cd models/ops
python setup.py build install
pip install -r requirements.txt
```
## Stage1
### Datasets
You can download icdar2015, icdar2013 and totaltext datasets from [this link](https://pan.baidu.com/s/1Rxuo9IqGcFAar8vL0TNpYg?pwd=evmg).

### Inference
The checkpoint can be download from [BaiduNetDisk](https://pan.baidu.com/s/1LdpX8rGu_tyWfHUmE79hlg?pwd=3cq5) (password:3cq5). Then modify ```data.test.img_rootdir``` and ```data.test.annotation_files``` in ```configs/finetune/ic15.yaml``` to your actual dataset path and run
```
python inference.py --config configs/finetune/ic15.yaml --resume /path/to/ckpt
```
to get the result of inference at ```inference```. You can also set ```inference.vis=True``` in config files if you want to visualize the results.

### Evaluation
Get the result file of inference, and then use the ground truth files downloaded from [Datasets link](https://pan.baidu.com/s/1Rxuo9IqGcFAar8vL0TNpYg?pwd=evmg) to evaluate:
```
python evaluation/evaluate_stage1.py --result inference/ic15_test.json --eval_gts dataset/eval_gts/ic15_test_gts.json
```

### Training
#### Pretrain
First modify ```data.train.annotation_files``` and ```data.train.img_rootdir``` in ```configs/pretrain/pretrain.yaml``` to your actual path, then run
```
python train.py --config configs/pretrain/pretrain.yaml
```
to start training. The outputs will be written to ```outputs/xxx```.

#### Finetune
First modify ```data.train.annotation_files``` and ```data.train.img_rootdir``` in ```configs/finetune/ic15.yaml``` to your actual path, then run
```
python train.py --config configs/pretrain/ic15.yaml --resume path/to/pretrain_ckpt
```
to start training. The outputs will be written to ```outputs/xxx```.

## Stage2
Waiting for update...

## Citation
```
@inproceedings{wu2025wecromcl,
  title={WeCromCL: Weakly Supervised Cross-Modality Contrastive Learning for Transcription-only Supervised Text Spotting},
  author={Wu, Jingjing and Fang, Zhengyao and Lyu, Pengyuan and Zhang, Chengquan and Chen, Fanglin and Lu, Guangming and Pei, Wenjie},
  booktitle={European Conference on Computer Vision},
  pages={289--306},
  year={2025},
  organization={Springer}
}
```