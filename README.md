# Head_Tracking_System

## Rol-Pitch-Yaw

<img alt="Resim açılamadı" width="500" height="600" src = "https://user-images.githubusercontent.com/59391291/150524230-227f9bc7-e34c-4999-8253-a75f8c2a96c9.png" />

## Algorithm

**MediPipe Face detection + Face Aligment + WHENet + RandomForestClassifier** <br>
We obtained Pitch and Yaw values with WHENet. Then we trained the Random Forest architecture using Pitch and Yaw values. With the trained model, where it faces (right-left-straight-down) was determined.

1) Detect Face with MediaPipe Face Detection <br>
2) Apply Face Aligment using MediaPipe Face Mesh Coordinates<br>
3) Obtain the pitch-yaw values ​​using the WHeNet model <br>
4) Classify where it's looking using the Random Forest model.<br>


**WheNet reference repo:** https://github.com/Ascend-Research/HeadPoseEstimation-WHENet <br>

## Predictions
![0](https://user-images.githubusercontent.com/59391291/159179413-96aec381-e6a6-4749-8773-30aa7fc28ddd.png)
![19](https://user-images.githubusercontent.com/59391291/159179415-a3f67fef-1799-4ad7-a6be-a3fcef631e19.png)
![70](https://user-images.githubusercontent.com/59391291/159179416-c69d16cc-5306-4dfe-963d-515e78b3b2f5.png)
![200](https://user-images.githubusercontent.com/59391291/159179417-53f8f807-c824-4ff4-8491-a79088df6406.png)
![406](https://user-images.githubusercontent.com/59391291/159179419-4da65a00-c7d1-443d-a37e-6ddcc470e4ac.png)
