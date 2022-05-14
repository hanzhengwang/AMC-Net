# AMC-Net

Code for paper "AMC-Net: Attentive modality-consistent network for visible-infrared person re-identification"


## Requirments:
pytorch: 1.6.0

torchvision: 0.2.1

numpy: 1.15.0

python: 3.7


## Dataset:
**SYSU-MM01**

**Reg-DB**


## Run:
### SYSU-MM01:
1. prepare training set
```
python pre_process_sysu.py
```
2. train model


To train a model with on SYSU-MM01 with a single GPU device 0, run:
```
python train.py --dataset sysu --lr 0.01 --drop 0.0 --trial 1 --gpu 0 --epochs 60 --w_hc 0.5 --per_img 8 
```

3. evaluate model(single-shot all-search)
```
python test.py --dataset sysu --lr 0.01 --drop 0.0 --trial 1 --gpu 0 --low-dim 512 --w_hc 0.5 --mode all --gall-mode single --model_path 'Your model path'
```

### Reg-DB:
1. train model
```
python train.py --dataset regdb --lr 0.01 --drop 0.0 --trial 1 --gpu 0 --epochs 60 --w_hc 0.5 --per_img 8
```

2. evaluate model
```
python test.py --dataset regdb --lr 0.01 --drop 0.0 --trial 1 --gpu 0 --low-dim 512 --w_hc 0.5 --model_path 'Your model path'
```

## Results:

 SYSU-MM01    [BaiduYun(code:8cg3)](https://pan.baidu.com/s/1GWVLxSOdYcBVX50Gl1xD-w)
 RegDB    [BaiduYun(code:nw81)](https://pan.baidu.com/s/1TTVZwh-mvzlgFWxj6defFA)



**Citation**

Most of the code are borrowed from https://github.com/98zyx/Hetero-center-loss-for-cross-modality-person-re-id. Thanks a lot for the author's contribution.

Please cite the following paper in your publications if it is helpful:
```
@article{wang2021amc,
  title={AMC-Net: Attentive modality-consistent network for visible-infrared person re-identification},
  author={Wang, Hanzheng and Zhao, Jiaqi and Zhou, Yong and Yao, Rui and Chen, Ying and Chen, Silin},
  journal={Neurocomputing},
  volume={463},
  pages={226--236},
  year={2021},
  publisher={Elsevier}
}

@article{zhu2020hetero,
  title={Hetero-center loss for cross-modality person re-identification},
  author={Zhu, Yuanxin and Yang, Zhao and Wang, Li and Zhao, Sai and Hu, Xiao and Tao, Dapeng},
  journal={Neurocomputing},
  volume={386},
  pages={97--109},
  year={2020},
  publisher={Elsevier}
}
```




