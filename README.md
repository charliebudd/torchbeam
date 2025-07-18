# TorchBeam
PyTorch native high-performance video streaming.

## Support
Currently, torchbeam has limited support in a number of areas. Contributions to expand this are welcome.

## Installs
Due to dependency issues with cuda and pytorch, installation currently requires compiling from source.
For this you will need both PyTorch and NVCC (Installed with the cuda toolkit). 

In order to ensure compatability, the NVCC cuda compilation major version must match the version used to build your PyTorch installation.
```
nvcc --version | grep "release" | sed 's/.*release \(.*\),.*/\1/'
python -c "import torch; print(torch.version.cuda)"
```
Note: only the major version is important, e.g. 12.0 and 12.6 are compatable.

Installing a version of PyTorch that matches your nvcc version can be done by specifiying the corresponding index url in the pip install command.
For example, the following command will install pytorch compiled with cuda 12.1...
```
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu121
```

Once your environment is setup, you are ready to install torchbeam. This must be done with the --no-build-isolation flag to ensure the compiled code is compatible with your pytorch installation.
```
pip install --no-build-isolation git+https://github.com/charliebudd/torchbeam.git
```