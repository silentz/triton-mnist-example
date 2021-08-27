# Triton inference server MNIST example

MNIST model inference example using
[NVIDIA Triton inference server](https://developer.nvidia.com/nvidia-triton-inference-server)
made as simple as possible. Example is implemented in Python programming languages. 

### Layout

```
├── client/
│   ├── client.py           # Python client
│   └── samples/            # Sample images
├── model/
│   ├── train.py            # Model training pipeline
│   └── requirements.txt    # Model requirements
└── repository/
    └── mnist/
        ├── 1/
        │   └── model.pt    # Traced mnist model
        └── config.pbtxt    # Triton model config
```

### Triton server

To run triton inference server, do following:

```bash
docker run -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD/repository:/models nvcr.io/nvidia/tritonserver:21.07-py3 tritonserver --model-repository=/models
```

## Triton client

Python triton image client. Only available on linux due
to triton client library limitations. Following steps are required
to run model:

1. Install dependencies:
```bash
pip3 install nvidia-pyindex
pip3 install tritonclient[all]
pip3 install opencv-python
```

2. Run client:
```bash
cd client
python3 client.py samples/0.png --url localhost:8000
```
