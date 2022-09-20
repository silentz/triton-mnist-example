import argparse
import cv2
import torch

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    args = parser.parse_args()

    # load model
    model = torch.jit.load('model.pt')
    model.eval()

    # read image
    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    image = torch.from_numpy(image)
    image = image.float()

    # create batch
    batch = image.unsqueeze(dim=0)

    # infer model
    with torch.no_grad():
        pred = model(batch)
        pred = torch.argmax(pred)
        print('Predicted digit:', pred.item())
