cd ..
git clone https://github.com/NVlabs/nvdiffrast.git

pip install ninja

pip3 install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128

cd ../nvdiffrast
python setup.py install
