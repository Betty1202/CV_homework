# CV_homework

## Environment
```shell
conda create -n opencv python=3.9
conda activate opencv
pip install opencv-python
pip install matplotlib
```

## SIFT
`SIFT/opencv.py` is the code from [url](https://zhuanlan.zhihu.com/p/157578594), `SIFT/opencv2.py` is the code from [url](https://blog.csdn.net/g11d111/article/details/79925827), and both of them use opencv.

`SIFT/01.png` and `SIFT/02.png` from [repo](https://github.com/rmislam/PythonSIFT.git), which has source code of SIFT.

And `main.py` is my homework.

## KLT
You need to run `KLT/camera_calibration.py` to get camera intrinsic parameters K. And put K into `main.py` for feature point tracking and camera position estimation.