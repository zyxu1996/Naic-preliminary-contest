# Naic-preliminary-contest
## Introduction
Rank 3th in the NAIC (National Artificial Intelligence Challenge) preliminary contest.
## Setting
* RTX2080Ti × 4
* Linux  
* Cuda10.1
## Usage
1. Install packages  
  * The code is tested on python3.5 and torch 1.1.0. (A higher version should be also work)  
  ```
  pip install -r requirements.txt  
  ```
2. Dataset  
  * Download the dataset on [NAIC](https://naic.pcl.ac.cn/frame/2)  
  * Please put dataset: train, image_A, image_B in folder `data`  
3. Train  
  * Run this command at the terminal:
  ```
  sh autorun_train.sh
  ```
  * You can change the python version in `autorun_train.sh`  
4. Test
  * The weights is available on line:  
  百度网盘：[https://pan.baidu.com/s/13efsRZx7qdqVCX5cJSf1-g](https://pan.baidu.com/s/13efsRZx7qdqVCX5cJSf1-g)  
  提取码：s7f2  
  Please download the weights and put it in folder `weights`
  * Testing command:  
  ```
  sh autorun_test.sh
  ``` 
5. Ensemble  
  * The best results in rank A and rank B, you can change the test path in folder `dataset` to get the results of dataset image_A and image_B.
  ```
  sh autorun_ensemble.sh
  ```
6. Results  
  * Stored in folder `results`
  ```
  sh zip.sh
  ```
  * Compressed to ZIP format. 
