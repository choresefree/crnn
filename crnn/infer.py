import argparse
import torch
from PIL import Image
from torchvision import transforms
from models import ResNetASTER
from utils import decode,  alphabet36, alphabet62, alphabet94


def infer(args):
    if args.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if args.alphabet == 36:
        alphabet = alphabet36
    elif args.alphabet == 62:
        alphabet = alphabet62
    else:
        alphabet = alphabet94
    # Define models and load pre-trained parameter.
    model = ResNetASTER(num_class=len(alphabet) + 1, with_lstm=True)
    state_dict = torch.load(args.pretrained_dir)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    # Load image for prediction.
    image = Image.open(args.img_dir).convert('RGB')
    w, h = image.size
    w_r = (w*args.height)//h
    img_tfs = transforms.Compose([
        transforms.Resize((args.height, w_r)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, ], [0.5, ])
    ])
    image = img_tfs(image).unsqueeze(0)
    image = image.to(device)
    output = model(image)

    # Get prediction.
    pre = decode(alphabet, output, method='greedy')[0]
    print(f'text prediction: {pre}')


def main():
    parser = argparse.ArgumentParser()
    # Required parameters.
    parser.add_argument('--img_dir', type=str, default='demo/hello kitty.png',
                        help='The path of the picture to be predicted.')
    parser.add_argument('--height', type=int, default=32,
                        help='Image height input to the network.')
    parser.add_argument('--alphabet', type=int, choices=[36, 62, 94], default=62,
                        help='Character dictionary.')
    parser.add_argument('--decode', type=str, choices=['greedy', 'beam'], default='greedy',
                        help='Decoding method.')
    parser.add_argument('--pretrained_dir', type=str, default='output/models/best.pth',
                        help='Preloaded models parameters.')
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='Whether to use gpu for acceleration.')
    args = parser.parse_args()
    infer(args)


if __name__ == '__main__':
    main()
