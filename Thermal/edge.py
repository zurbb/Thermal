from main import *
import cv2
import numpy as np
from matplotlib import pyplot as plt


p = ImageProcess(  r"C:\Users\zur99\Pictures\flir original\b.JPG")


def hilight(image, edges):
    x, y = edges.shape
    for pix_x in range(x):


        for pix_y in range(y):
            if edges[pix_x][pix_y] == 255:
                image[pix_x][pix_y] = 0

    return image


m=
a = [(m-1.5,"purple"), (m-1, "blue"), (m-0.5, "yellow"), (m, "white")]



def make_layer(layer_arr, process_image):
    p = process_image
    pos =0
    for x in layer_arr:
        temp ,color = x

        i = p.filter_thermal_by_celsius(temp)
        edges = cv2.Canny(i, 100, 200)
        img = p.marker_visual_by_celsius(temp, color)
        cv2.putText(img, "{} > {}".format(color ,temp), (10 , 20 + pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color_bgr(color) )
        hilight(img, edges)
        pos = pos+ 30

make_layer(a ,p)
p.show_visual()
cv2.waitKey(0)


















