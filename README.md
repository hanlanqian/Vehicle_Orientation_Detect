# Vehicle_Orientation_Detect

This is a two-stage vehicle orientation detection method.

## Simple Usage

- calibration stage

```shell
python app.py --source ${video path} --engine ${tensorRT engine} --caliFlag True
```

- detect stage

```shell
python app.py --source ${video path} --engine ${tensorRT engine} --caliFlag False --calibration ${calibration path}
```

## Workflow

### Camera Calibration

- Vanishing Points Detect
    1. first VP
        - Get the ROI of image by vehicle detect
        - track the vehicles by KLT tracker(**optical flow method**)
        - push all tracks gathered in last step to **DiamondSpace** to get the intersections(which is considered as **
          First VP**)
        - Get the first vanishing point by voting algorithm
    2. second VP
       - 

### Orientation Detection

- KeyPoint Detect
    - Based on an 2D human position estimation network project[openpifpaf](https://github.com/openpifpaf/openpifpaf)
    - we get a pair of keypoints which can represent the orientation of vehicle(such as car lights)

- Get Bird-View image plane