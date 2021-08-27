import argparse
import cv2
import numpy as np
import tritonclient.http as tritonclient


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='path to digit image')
    parser.add_argument('--url', type=str, default='localhost:8000', help='triton server url')
    args = parser.parse_args()

    image = cv2.imread(args.path)
    image = np.mean(image, axis=2)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)

    client = tritonclient.InferenceServerClient(args.url)
    model_name = 'mnist'

    inputs = [
            tritonclient.InferInput('input__0', [1, 28, 28], "FP32"),
        ]

    outputs = [
            tritonclient.InferInput('output__0', [1, 10], "FP32"),
        ]

    inputs[0].set_data_from_numpy(image, binary_data=True)

    model_out = client.infer(model_name, inputs, outputs=outputs)
    model_out = model_out.as_numpy('output__0')[0]
    digit = np.argmax(model_out)
    print('Predicted label:', digit)
