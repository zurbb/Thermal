import pickle
import cv2

with open("test.text", "rb") as fp:
    b= pickle.load(fp)

while True:
    for x in b[0:-35]:
        cv2.imshow("test", x)
        cv2.waitKey(600)