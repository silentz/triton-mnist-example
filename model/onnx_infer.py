import onnxruntime as ort
import numpy as np
import cv2
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    args = parser.parse_args()

    # read image
    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)

    ort_sess = ort.InferenceSession('model.onnx')
    outputs = ort_sess.run(None, {'input_0': image})

    # Print Result
    predicted = outputs[0][0].argmax(0)
    print(f'Predicted: "{predicted}"')
