from warnings import filterwarnings
from whenet import WHENet
import cv2
import os 
from demo import process_detection
whenet = WHENet(snapshot="WheNet model path ...")
video_path = "..."



def video_maker(path,video_name):
    images = [img for img in os.listdir(path) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(path+video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(path, image)))


i=0
cap = cv2.VideoCapture(video_path)
while True:
    ret, frame = cap.read()
    if ret: 
        frame = process_detection(whenet, frame)
        cv2.imwrite(f"./results/{i}.png", frame)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        
        break
    i+=1

cap.release()
cv2.destroyAllWindows()
video_maker("./results/","video.avi")