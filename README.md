## HSNet: A Hybrid Semantic Network for Polyp Segmentation

We investigate a hybrid semantic network (HSNet) that adopts both the advantages of transformer and convolutional neural networks (CNN), aiming to improve polyp segmentation. The HSNet contains a cross-semantic attention module (CSA), a hybrid semantic complementary module (HSC), and a multi-scale prediction module (MSP). 

The details of this project are presented in the following paper:

[HSNet: A Hybrid Semantic Network for Polyp Segmentation [CIBM'22]](https://pdf.sciencedirectassets.com/271150/1-s2.0-S0010482522X00097/1-s2.0-S0010482522008812/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEHwaCXVzLWVhc3QtMSJHMEUCIG%2BROVYdGoXsTd2GKEF9DuujbApi4x9XsFMH33fJVg4%2BAiEAmuMWtsA8rS0nqpeC1%2FGn5JLxxbAWo9kRje%2Bt3KOXTYEquwUI1P%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDLkzh3m0hcGxlKcLYyqPBfIl4XzIPW3Vf7tCcjvQQS9Weow%2Fhb3%2B5aCq3VonoZNs2Fr9oZMuyFnjk30DCw%2BvdUhMe54qopXdgOYU0NRqJo5hGMMF1CR4j5TNvyIbcATnq5O2Kr%2BIVjucqIHYFO4srVnsb6OJXfh9KKV3D8OTCA34XUpehTRkxu2DZozUoM3DI7eRtjUE1p3L3Hcn%2BQZqMFFbMICrXNyTnBeqbIqu6ygFa23opNhJWhkNmMzLw%2FMqdC%2B9tetKpA9JD09%2BprVCkGqYM6t6S8HjDZfR3JBOZHpIMTbJmyJE6JT77kza6WKgQ0bYMnv%2BbFiMwpJFqbOuKwfowHul11kPrpX7%2FBNTSFUUbr56CKzzc0Fly0Ru%2BWMQul3foEvZ4qnujmTmOVq7w1CHgQXlJXqkLst0wcIliFDrpphwMN%2BzKVj9ZmDyDVfNAqNpcTAC0y%2FA20LGuvASOTmQKI0Gw2jWeU66jgUZFCRSy9Mp3VylKO3NzD7h%2BM5vp5zyhgLgGtbj1WrkNAuEbTzfK%2FDFJh9VdhI7Fym2djpZ9%2Fn%2B7eN0wy%2Bc%2Fc%2FRBXxxFxPFoBR1ZYXWhuhOfh%2FWnNYZ5gIwt6O%2B2V5K4GGSi10TT4pQCugUGygxq3t8XLQ5n9LZZq3kM0Ea9x7CuVQoUpjwJFLV2JYMJnY4VNAmsXPsgNiZtY%2FmRq1If0Z71EVgjfiHVBTd%2BDKzmQ3k6gRANtI9VJqv4L4cggqlZMvtZ1hkGCGXMUD99iPniMq0smjVf8567RQIeSXVq0y7j7%2FY%2BEEk5mdSYHwDSrC6Dlg8DhuZNPb8XgWnm82dPwBRBLePGYjfttltHezjPVgg0UzPNdu2a1kjBBeIkT7XjoRueV4PPL9n8PO7LHuKFnGpfIkwotnQpAY6sQH6WaHyKgcTsTg9Z1hYSVh1UADz%2F3pItuFiCuwUnqdwktYvPSvxYZYiVB0IwQU7BMPZvfqujnLct4%2B2RaefWb%2Fuaq9xOxe5k8tjlni7lSsjgkDXxccQCpOu0XT5GBGBkYTFaQpqYFB8H3%2BzyUPcrLhPwj%2FmDQerbu2eMuYQXV1iHmOJv2vysUmc3GeMqPCkZRwXVkeFep9%2Bpg5e37HTiwnLwRrKAkElrzSMXXgJlnL91KE%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230622T130453Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYQCED4ONZ%2F20230622%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=11bc501bfd9aee076e8324b48a0c2478afd2e77be595132e04de5a72b7f7b674&hash=67dc160ddd80275cc75698099a1c3c47acfdc493a2ea6415a2fb3166ef6cb575&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0010482522008812&tid=spdf-1e6adc1b-1bc4-4e9e-8b2c-0964791bcafb&sid=7b491883798a934ed62a1db4d2cda00d61a4gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0e0852050c5b0754520f&rr=7db4b31a1aeb8b81&cc=hk) 
<br>**[Wenchao Zhang](https://www.researchgate.net/profile/Wenchao-Zhang-30), [Chong Fu](https://scholar.google.com/citations?user=xq76xEMAAAAJ&hl=zh-CN), [Yu Zheng](https://github.com/yuzhengcuhk), Fangyuan Zhang, Yanli Zhao, and [Chiu-Wing Sham](https://scholar.google.com/citations?user=b-hQ_U8AAAAJ&hl=en)**<br>


## Usage 
### Setup 
```
Python 3.8
Pytorch 1.7.1
torchvision 0.8.2
```
### Dataset 
Download the training and test datasets and move them into `./dataset/`, see [Google Drive](https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1OBVivLJAs9ZpnB5I2s3lNg) [code:dr1h].

### Pre-trained model 
Download the pre-trained model from [Google Drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1Vez7iT2v_g7VYsDxRGE8HA) [code:w4vk], and then put it in the `./pretrained_pth`  folder for initialization. 

### Train the model 
Clone the repository
```
git clone https://github.com/baiboat/HSNet.git
cd HSNet 
bash train.sh
```

### Test the model
```
cd HSNet 
bash test.sh
```

### Evaluate the trained model 

```
cd HSNet 
python Eval.py
```


### Well-trained model 
[Baidu Drive](https://pan.baidu.com/s/11gbrzpmV82oYXFr09R7G-A) [code:hsnt] and put the model in directory `./model_pth`.

##  License
The source code is free for research and education use only. Any commercial use should get formal permission first.

Any advice is welcomed ^.^; please get in touch with **sylgzwc@163.com** or pull the question.

## Acknowledgement
Thanks [PraNet](https://github.com/DengPingFan/PraNet), [EAGRNet](https://github.com/tegusi/EAGRNet), [MSEG](https://github.com/james128333/HarDNet-MSEG) and [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) for serving as building blocks of HSNet.

## Citation

If you find our work/code interesting, welcome to cite our paper >^.^<

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
