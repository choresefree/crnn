# Crnn: Baseline for scene text recognition frame with resnet backbone
## Background
### Crnn is a classic end-to-end framework for scene text recognition and the paper can be loaded from https://arxiv.org/abs/1507.05717. This project uses resnet in Aster network as the backbone of CNN to build a basic baseline of scene text recognition. After training on the optical English character recognition training set, it can achieve good accuracy on the test set.
---
## Usage
### 1 This project gives a baseline of training and validation with ocr standard data set in lmdb format. Websites for downloading training set and test set in LMDB format can be found from https://github.com/FangShancheng/ABINet.
After downloading datasets, you can start training and validation by using instruction: 

    python train.py --train_dir path --val_dir path

Or use several datasets at the same time: 

    python train.py --train_dir path1|path2 --val_dir path1|path2|path3

### 2 Or you can train on your own dataset by use SimpleDataset in utils/dataset.py.
The csv file content format should be:

    img_name,label
    data/xzy/datasets/train/1.jpg,sorry
    data/xzy/datasets/train/2.jpg,mother
    ......

### 3 Inference on single image has been given in infer.py.
This project provices two methods for decoding: greedy and beam search. You can finish inference by using instruction:

    python infer.py --img_dir path --pretrained_dir path
    
## Checkpoint
### This project provides a model with several iterations to prove it is convergent, and the weight can be downloaded from https://pan.baidu.com/s/11-0mw2yauhAr5iNatwS_lA , code: ag34
The accuracy of the weight is as follows:

| Dataset  | IC03 | IC13 | IIIT5k |              
| :---: | :---: | :---: | :---: |
| Number  | 860 | 857 | 5000 |                     
| Accuracy  | 69.2 | 63.5 | 48.2|
