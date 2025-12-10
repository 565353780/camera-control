cd ..
git clone https://github.com/Dao-AILab/flash-attention.git

pip3 install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128

cd flash-attention
git checkout v2.8.3

pip install ninja packaging

MAKEFLAGS="-j$(nproc)" python setup.py build
python setup.py install
