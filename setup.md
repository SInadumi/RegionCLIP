### setup by poetry
```
poetry init
poetry install
python setup.py develop # XXX: no longer works "poetry install"
```

### setup by miniconda
```
conda install pytorch=1.9.0 torchvision=0.10.0 pytorch-cuda=12.1 -c pytorch -c nvidia
source activate regionclip
pip install setuptools==59.0.1
python -m pip install -e .
pip install opencv-python timm diffdist h5py scikit-learn ftfy
pip install git+https://github.com/lvis-dataset/lvis-api.git
```
