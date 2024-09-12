This solution is developed based on the following repository:

https://github.com/leeyeehoo/CSRNet-pytorch.git

The main process consist of two parts:
1. Training is base on ShanghaiTech dataset
2. The model used is CSRNet

I only trained for a short period of time and best MAE is around 54. Only part_A dataset is used in training.
The initial training is done by using the following command:

python train.py part_A_train.json part_A_val.json 0 0

To restart the training, the following command is used:

python train.py part_A_train.json part_A_val.json 0 0 --pre .\0checkpoint.pth.tar
(For fast training, CUDA is used)

Subsequently, I extracted the video frame by frame.
Due to the way the question is set up and time-constraint, two frames were selected to construct a paranorma scene
for estimating the crowd size on bridge.

Then, I performed testing using the following command:

python .\test.py --pre .\0checkpoint.pth.tar
(Done on CPU, easy to repredict. But the trained model cannot be upload due to github filesize limit)

Final output of the prediction:

=> loaded checkpoint '.\0checkpoint.pth.tar' (epoch 131)
begin test
('./test_image/video_frame350.png',): prediction: 873.2084350585938
('./test_image/video_frame700.png',): prediction: 570.5108642578125

The final estimation is approximately 1443 people in the crowd. From the two selected images,
there are less people in video_frame350, but the estimated number is higher than the predicted number
from video_frame700. Thus, it is obvious the model still have rooms for improvements. There are
several factors that can impact the model performance:
1. The time for training the model is short.
2. The resolution has been compromised when downloading the video.
3. The density of crowd is high for video_frame700. (Also, also apply to many other frames.)

Alternative solution:

Instead of performing prediction on only two frames, conducting predictions continuously to all frames and integrate them together is also feasible to solve this problem. However, I need to solve an additional issue here, which is to identify/tracking duplicated crowd between frames. Unfortunately I have to chose not to proceed in this way due to the short period of time.
A quick layout on how to integerate continuesly:
1. get the predicted density map and crowd set $X_0$ from frame 0.
2. get the predicted density map and crowd set $X_1$ from frame 1.
3. use the center region of frame 0, identify its location in frame 0, so the shift between frame 0 and frame 1 can be calculated (dx, dy)
4. use the predicted density map from frame 0, find the region that is not in frame 1 based on calculated (dx, dy), and count of crowd that is not belong to frame 1 ($X_0 - X_1$).
5. use the predicted density map from frame 1, find the region that is not in frame 0 based on calculated (dx, dy), and count of crowd that is not belong to frame 0 ($X_1 - X_0$).
6. The crowd in the overlapped region can be calculated as $X_0 \cap X_1 = \frac{X_0 - (X_0 - X_1)}{2} + \frac{X_1 - (X_1 - X_0)}{2}$. (note: these two can be different based on prediction from the 2 frames, thus taking average)
7. And the overall region crowd can be calculated as $X_0 \cup X_1 = X_0 \cap X_1 + (X_0 - X_1) + (X_1 - X_0)$
8. Extend this to multiple frames.
