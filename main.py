import gdown
from train import train_lipnet
import argparse
import json
import os


def get_parser():
    parser = argparse.ArgumentParser(description='Set up the training parameters.')

    parser.add_argument('--epoch', dest='epoch', type=int, default=300, help='Number of epochs for training.')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=256, help='Size of hidden layers.')
    parser.add_argument('--model', dest='model', type=str, default='lipnet-lstm', help='Model file.')
    parser.add_argument('--batch', dest='batch', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--workers', dest='workers', type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument('--device', dest='device', type=str, default='cuda', help='Computing device')
    
    return parser

def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)

if __name__ == '__main__':

    # Get the parser with argument definitions
    parser = get_parser()

    opts = parser.parse_args()

    output = 'data.zip'
    url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'

    if not os.path.isfile(output):
        print("Initializing data ingestion...")
        gdown.download(url, output, quiet=False)
        gdown.extractall('data.zip')

    print("Data Installed...\n Saved in data/")

    print("Training...")
    print("~"*200)
    summary = train_lipnet(opts=opts)
    summary["opts" : opts]
    print("~"*200)
    print("Finished training")
    save_json(summary, "summary.json")