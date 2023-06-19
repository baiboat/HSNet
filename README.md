# HSNet

The code for the paper entitled "HSNet: A Hybrid Semantic Network for Polyp Segmentation."
By Wenchao Zhang, Chong Fu, Yu Zheng, Fangyuan Zhang, Yanli Zhao, and Chiu-Wing Shamg.


## 1. Introduction

Automatic polyp segmentation can help physicians to effectively locate polyps (a.k.a. region of interests) in clinical practice, in the way of screening colonoscopy images assisted by neural networks (NN). However, two significant bottlenecks hinder its effectiveness, disappointing physicians’ expectations. 1) Changeable polyps in different scaling, orientation, and illumination, bring difficulty in accurate segmentation. 2) Current works building on a dominant decoder-encoder network tend to overlook appearance details (e.g., textures) for a tiny polyp, degrading the accuracy to differentiate polyps. For alleviating the bottlenecks, we investigate a hybrid semantic network (HSNet) that adopts both advantages of Transformer and convolutional neural networks (CNN), aiming at improving polyp segmentation. Our HSNet contains a cross-semantic attention module (CSA), a hybrid semantic complementary module (HSC), and a multi-scale prediction module (MSP). Unlike previous works on segmenting polyps, we newly insert the CSA module, which can fill the gap between low-level and high-level features via an interactive mechanism that exchanges two types of semantics from different NN attentions. By a dual-branch structure of Transformer and CNN, we newly design an HSC module, for capturing both long-range dependencies and local details of appearance. Besides, the MSP module can learn weights for fusing stage-level prediction masks of a decoder. Experimentally, we compared our work with 10 state-of-the-art works, including both recent and classical works, showing improved accuracy (via 7 evaluative metrics) over 5 benchmark datasets, e.g., it achieves 0.926/0.877 mDic/mIoU on Kvasir-SEG, 0.948/0.905 mDic/mIoU on ClinicDB, 0.810/0.735 mDic/mIoU on ColonDB, 0.808/0.74 mDic/mIoU on ETIS, and 0.903/0.839 mDic/mIoU on Endoscene. 


## 2. Usage:
### 2.1 Recommended environment:
```
Python 3.8
Pytorch 1.7.1
torchvision 0.8.2
```
### 2.2 Data preparation:
Downloading training and testing datasets and move them into ./dataset/, which can be found in this [Google Drive](https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1OBVivLJAs9ZpnB5I2s3lNg) [code:dr1h].


### 2.3 Pretrained model:
You should download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1Vez7iT2v_g7VYsDxRGE8HA) [code:w4vk], and then put it in the './pretrained_pth' folder for initialization. 

### 2.4 Training:
Clone the repository:
```
git clone https://github.com/baiboat/HSNet.git
cd HSNet 
bash train.sh
```

### 2.5 Testing:
```
cd HSNet 
bash test.sh
```

### 2.6 Evaluating your trained model:

```
cd HSNet 
python Eval.py
```


### 2.7 Well trained model:
[Baidu Drive](https://pan.baidu.com/s/11gbrzpmV82oYXFr09R7G-A) [code:hsnt] and put the model in directory './model_pth'.

## 3. Acknowledgement
We are very grateful for these excellent works [PraNet](https://github.com/DengPingFan/PraNet), [EAGRNet](https://github.com/tegusi/EAGRNet), [MSEG](https://github.com/james128333/HarDNet-MSEG) and [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT), which have provided the basis for our framework.

## 4. FAQ:
If you want to improve the usability or any piece of advice, please feel free to contact me directly (sylgzwc@163.com).

## 5. License
The source code is free for research and education use only. Any comercial use should get formal permission first.

## 6. BibTeX

If you find our work and this repository useful. Please consider giving a star and citation;.

```bibtex
@article{zhang2022hsnet,
  title={HSNet: A hybrid semantic network for polyp segmentation},
  author={Zhang, Wenchao and Fu, Chong and Zheng, Yu and Zhang, Fangyuan and Zhao, Yanli and Sham, Chiu-Wing},
  journal={Computers in Biology and Medicine},
  volume={150},
  pages={106173},
  year={2022},
  publisher={Elsevier}
}
```
