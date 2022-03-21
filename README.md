# Vehicle_Orientation_Detect

- 这是一个针对交通监控场景下精确检测车辆朝向的项目

## Camera Calibration

- 消失点检测
    1. first VP
        - 车辆目标检测(SSD(backbone:Resnet50))获取感兴趣区域
        - KLT车辆追踪获取轨迹集
        - 轨迹集通过级联霍夫变换得到所有交点
        - 通过投票算法得到最终的第一个消失点
    2. second VP