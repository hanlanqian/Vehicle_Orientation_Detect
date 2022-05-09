# 车辆方向检测

## 相机标定

- 消失点检测
    1. 第一消失点
        - Get the ROI of image by vehicle detect
        - track the vehicles by KLT tracker(**optical flow method**)
        - push all tracks gathered in last step to **DiamondSpace** to get the intersections(which is considered as **First VP**) 
        - Get the first vanishing point by voting algorithm
    2. second VP
       - 

## Orientation Detection

- KeyPoint Detect
  
- Get Bird-View image plane