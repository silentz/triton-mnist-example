## MNIST model

Traced pytorch model and training pipeline.

#### How to train model from scratch

1. Create python virtual environment:
```bash
virtualenv --python=python3 venv
```

2. Activate virtual environment:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train and trace model:
```
python train.py
```

5. Place traced model in model repository:
```
cp model.pt ../repository/mnist/1/model.pt
```
