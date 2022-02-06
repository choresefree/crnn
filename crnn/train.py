import argparse
import os
import random
import re
import torch
import numpy as np
from torch import nn, optim
from torch.backends import cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import decode, cal_levenshtein, \
    alphabet36, alphabet62, alphabet94, \
    AlignCollate, LmdbDataset, ImageConcatDataset
from models import ResNetASTER


def train(args):
    if args.alphabet == 36:
        alphabet = alphabet36
    elif args.alphabet == 62:
        alphabet = alphabet62
    else:
        alphabet = alphabet94

    # Load datasets from files in lmdb format.
    dirs = args.train_dir.split('|')
    train_dataset = [LmdbDataset(p, alphabet) for p in dirs]
    train_dataset = ImageConcatDataset(train_dataset)
    # If you want to use SimpleDataset, change the code here.
    # # Load datasets from files in simple format.
    # train_dataset = SimpleDataset(args.train_dir, alphabet)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=AlignCollate(*args.img_size),
                              pin_memory=True
                              )
    train_number = len(train_dataset)
    print(f'train sample: {train_number}')
    dirs = args.val_dir.split('|')
    val_dataset = [LmdbDataset(p, alphabet=alphabet) for p in dirs]
    val_dataset = ImageConcatDataset(val_dataset)
    # val_dataset = SimpleDataset(args.val_dir, alphabet)
    val_loader = DataLoader(val_dataset,
                            batch_size=16,
                            shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=AlignCollate(*args.img_size),
                            )
    val_number = len(val_dataset)
    print(f'val sample: {val_number}')

    # Define models and load pre-trained parameter.
    model = ResNetASTER(num_class=len(alphabet) + 1, with_lstm=True)
    if args.pretrained_dir is not None:
        print(f'load weight from {args.pretrained_dir}')
        state_dict = torch.load(args.pretrained_dir)
        model.load_state_dict(state_dict)
    model = model.cuda()

    # Define hyper-parameters.
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).cuda()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=10, last_epoch=-1, gamma=0.1)
    epochs = args.epoch

    # Models training and validation.
    best_val_acc = 0
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for images, labels, _, label_lens in tqdm(train_loader, desc='epoch %s training' % (epoch + 1)):
            model.train()
            images = images.cuda()
            # Forward.
            output = model(images).log_softmax(2).permute(1, 0, 2)
            pre_lens = torch.IntTensor([output.size(0)] * len(label_lens))
            loss = criterion(output.cpu(), labels, pre_lens, label_lens)
            epoch_loss += loss.item()
            # Backward.
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            step += 1
            if step % args.save_step == 0:
                #  Save the latest model.
                torch.save(model.state_dict(), f'{args.output_dir}/latest.pth')
            if step % args.val_step == 0:
                model.eval()
                val_acc = 0
                val_dist = 0
                with torch.no_grad():
                    for val_images, _, val_texts, _ in tqdm(val_loader, desc='models val'):
                        images = val_images.cuda()
                        output = model(images)
                        pres = decode(alphabet, output.cpu(), method=args.decode)
                        for text, pre in zip(val_texts, pres):
                            text = re.sub('[^0-9a-zA-Z]+', '', text).lower()
                            pre = re.sub('[^0-9a-zA-Z]+', '', pre).lower()
                            if text == pre:
                                val_acc += 1
                            else:
                                val_dist += cal_levenshtein(text.lower(), pre.lower()) / (max(len(text), len(pre)))
                    #  Save the best model on validation datasets.
                    print(f'val_acc: {val_acc / val_number} val_dist: {val_dist / val_number}')
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(model.state_dict(), f'{args.output_dir}/best.pth')
        print(f'epoch {epoch + 1} train_avg_loss: {epoch_loss / len(train_loader)}')
        scheduler.step()


def main():
    assert torch.cuda.is_available(), 'Gpus could not be used.'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()

    # Required parameters.
    parser.add_argument('--train_dir', type=str, required=True,
                        help='File path of training datasets (Different paths are separated by |).')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='File path of validation datasets (Different paths are separated by |).')
    parser.add_argument('--save_step', type=int, default=2000,
                        help='Steps interval for models saving.')
    parser.add_argument('--val_step', type=int, default=10,
                        help='Steps interval for models validation.')
    parser.add_argument('--alphabet', type=int, choices=[36, 62, 94], default=62,
                        help='Character dictionary.')
    parser.add_argument('--decode', type=str, choices=['greedy', 'beam'], default='greedy',
                        help='Decoding method.')
    parser.add_argument('--pretrained_dir', type=str, default=None,
                        help='Preloaded models parameters.')
    parser.add_argument('--output_dir', type=str, default='output/models',
                        help='Path to save models.')
    parser.add_argument('--img_size', type=tuple, default=(32, 100),
                        help='Input image size (Height must be 32 while 100 is recommended for English and 320/640 is '
                             'more suitable for Chinese)')
    parser.add_argument('--clip_grad', type=float, default=10.0,
                        help='Truncate the gradient to prevent the training instability and even abnormality caused '
                             'by too large back propagation gradient.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='The initial learning rate for SGD.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight_decay for SGD.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of forward propagation and back propagation of data sets.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of each training batch.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='The number of sub processes created during data loading and it should be set to 0 in '
                             'Windows system.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed setting.')
    args = parser.parse_args()

    # Seed setting.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    train(args)


if __name__ == '__main__':
    main()
