# VeIT-CV

Real Time Pose Stream to Unity Steps:

1. use stereoCalibration.py to calibrate two Webcams, follow the steps in main comment and uncomment commands in each step the calibrated camera parameters will be in the cameraCal/camera_parameters folder P.s. If something wrong, check the savefolder path in utils.py
2. use mzh_RealTimePoseStreamToUnity.py to visualize the keypoints and stream to unity P.s. Change keypoints accordingly The image is fliped to change the orientation in unity Comment or uncomment visualize_3d(frame_p3ds, ax) to see in 3D (not necessary since need to change visualize_3d func) P.s. To stream to unity, change UDP ID and port
