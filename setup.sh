cd ..
git clone --depth 1 https://github.com/NVlabs/nvdiffrast.git

pip install ninja trimesh open3d viser

pip3 install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu124

cd nvdiffrast
python setup.py install
