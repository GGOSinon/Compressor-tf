# Compressor-tf
Tensorflow compressor for high resolution images.

# Requirements
numpy==1.11.0

scikit_image==0.13.1

Pillow==5.1.0

tensorflow==1.8.0

You can install these libraries by using 'requirements.txt' in the project.

# Installation

# Unix(Linux, Mac)
1. To install this project, type 'git clone https://github.com/GGOSinon/Compressor-tf/' in terminal

2. To install requirements, type 'pip install -r requirements.txt' in terminal.

# Windows(PyCharm)
1. To install the project, you can download the entire project in this page.

2. To install requirements, you must install these libraries manually in PyCharm.(pip 10.1.0 recommended)

# Using GPU
If you want to use GPU, then several steps must be added.
1. Install tensorflow-gpu==1.7.0
2. Install CUDA 9, cudnn 9.0.0.

# Usage
You can read informations by typing 'python Tester_fin.py --help'.

# Compression
python Tester_fin.py --path=image_path --mode=com --qf=qf

# Decompression
python Tester_fin.py --path=decompress_dir_path --mode=dec --new_path=new_image_path

# Testing
python Tester_fin.py --path=image_path --mode=all --new_path=new_image_path --qf=qf
