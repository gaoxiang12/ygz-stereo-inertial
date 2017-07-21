# YGZ-Stereo-Inertial
This is YGZ-stereo-inertial SLAM, a stereo inertial VO code. It is designed for stereo and stereo-inertial sensor modules like vi-sensor. It uses a LK optical flow as front-end and a sliding window bundle adjustment as a backend. Feel free to try it in datasets and your own sensors. 

The code is experimental and I can't guarantee  its performance in all cases.

# Dependency
If you are using ubuntu, just type "./install_dep.sh" to install all the dependencies except pangolin.

- Pangolin (for visualization): https://github.com/stevenlovegrove/Pangolin 
- Eigen3: sudo apt-get install libeigen3-dev
- g2o: sudo apt-get install libcxsparse-dev libqt4-dev libcholmod3.0.6 libsuitesparse-dev qt4-qmake 
- OpenCV: sudo apt-get install libopencv-dev
- glog (for logging): sudo apt-get install libgoogle-glog-dev

# Compile
run "./generate.sh" to compile all the things, or follow the steps in generate.sh

# Examples
You can put stereo or stereo-imu data into ygz-stereo, for example the EUROC dataset 
(http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets). We provide the pure stereo vision and stereo-inertial vision for EUROC. Run the pure vision examples by typing:

```
bin/EurocStereo ./examples/EurocStereo.yaml
```

to run the pure vision mode. Don't forget to specify the dataset directory in the yaml config file first. Also, to run visual-inertial mode, type: 
```
bin/EurocStereoVIO ./examples/EurocStereoVIO.yaml
```

to run the stereo VIO case.

# Other things
YGZ-stereo is more robust than the previous YGZ-ORB-SLAM (and the code is also more clear). In EUROC it can pass the test of all MHxx and V101, V201, V202. For difficult cases it may still fail. As I will go to TUM soon, my future work may be a visual-inertial DSO.

YGZ stands for Yi-Guo-Zhou (a port of porridge, a group of mess) because it contains feature method, direct method and imu things.

The Note.md is a file of develop records.

Contact me (gaoxiang12@mails.tsinghua.edu.cn) or Wang Jing (https://github.com/jingpang) for commercial use.

Thanks the following companies/people for financial support:
- Usens凌感科技
- Hyperception 远形时空
- Qfeeltech 速感科技
- 天之博特
- 视辰信息科技

