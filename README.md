# Vehicle_Orientation_Detect

## Camera Calibration

- Vanishing Points Detect
    1. first VP
        - Get the ROI of image by vehicle detect
        - track the vehicles by KLT tracker(**optical flow method**)
        - push all tracks gathered in last step to **DiamondSpace** to get the intersections(which is considered as **First VP**) 
        - Get the first vanishing point by voting algorithm
    2. second VP
       - 

## Orientation Detection

- KeyPoint Detect
  
- IPM