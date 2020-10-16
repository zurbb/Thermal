import flyr
from matplotlib import cm
import flirimageextractor

import numpy as np
import cv2
from colour import Color
import pickle








def hilight(image, edges):
    x, y = edges.shape
    for pix_x in range(x):


        for pix_y in range(y):
            if edges[pix_x][pix_y] == 255:
                image[pix_x][pix_y] = 0

    return image

def color_bgr(color):
    c = Color(color).rgb
    c = tuple(int(255 * x) for x in c)
    return c[::-1]


def filter_image (thermal_values,image, temp,color):

    tv = thermal_values
    image1 = image
    x, y = tv.shape
    for pix_x in range(x):
        if temp is None:
            continue

        for pix_y in range(y):
            if tv[pix_x][pix_y] < temp:
                image1[pix_x][pix_y] = color_bgr(color)
    return image1


def marker_image(thermal_values, image, temp, color):

    tv = thermal_values
    image1 = image
    x, y = tv.shape
    for pix_x in range(x):
        if temp is None:
            continue

        for pix_y in range(y):
            if tv[pix_x][pix_y] > temp:
                image1[pix_x][pix_y] = color_bgr(color)
    return image1


def mix_image(thermal_values, visual, thermal, temp):

    tv = thermal_values
    v = visual
    th = thermal

    x, y = tv.shape


    for pix_x in range(x):
        if temp is None:
            continue

        for pix_y in range(y):
            if tv[pix_x][pix_y] > temp:
                v[pix_x][pix_y] = th[pix_x][pix_y]
    return v







class ImageProcess:

    def __init__(self,flir_path):
        flir = flirimageextractor.FlirImageExtractor()
        flir.process_image(flir_path, RGB=True)
        thermogram = flyr.unpack(flir_path)
        thermal_values = np.array(thermogram.celsius)
        (y, x) = thermal_values.shape
        thermal = cv2.imread(flir_path)
        visual = flir.get_rgb_np()
        visual = cv2.resize(visual, (x, y))
        thermal = cv2.resize(thermal, (x, y))


        self.flir_path = flir_path
        self.visual = visual
        self.thermal = thermal
        self.thermal_values = thermal_values

    def reset(self):
        flir = flirimageextractor.FlirImageExtractor()
        flir.process_image(self.flir_path, RGB=True)
        thermogram = flyr.unpack(self.flir_path)
        thermal_values = np.array(thermogram.celsius)
        (y, x) = thermal_values.shape
        thermal = cv2.imread(self.flir_path)
        visual = flir.get_rgb_np()
        visual = cv2.resize(visual, (x, y))
        thermal = cv2.resize(thermal, (x, y))


        self.visual = visual
        self.thermal = thermal
        self.thermal_values = thermal_values



    def show_visual(self):

        cv2.imshow("visual", self.visual)
        cv2.waitKey(0)

    def show_thermal(self):

        cv2.imshow("thermal",self.thermal)
        cv2.waitKey(0)

    def filter_visual_by_celsius(self, temp= None, color="black"):

        img = filter_image(self.thermal_values, self.visual, temp, color)

        return img

    def filter_thermal_by_celsius(self, temp=None, color="black"):
        img = filter_image(self.thermal_values,self.thermal, temp, color )

        return img

    def filter_visual_by_percent(self, percent=100, color="black"):
        '''
        filter pixel by percentage of the highest temperature
        :param percent: int, defines the percentage of pixels to filter by temperature height, pixels below this
         percentage will not be displayed
        :param color: defines the color that the filtered pixels will display
        :return: image after processing
        '''

        temp_sort=[]
        for x in self.thermal_values:
            for y in x:
                temp_sort.append(y)

        temp_sort.sort()
        indx = int(len(temp_sort)*percent/100)
        low_temp = temp_sort[-indx]
        img = filter_image(self.thermal_values,self.visual, low_temp, color)
        return img

    def marker_visual_by_celsius(self, temp ,color ="green"):
        img = marker_image(self.thermal_values,self.visual, temp, color )
        return img
    def marker_visual_by_precent(self, percent=100, color="green"):
        temp_sort = []
        for x in self.thermal_values:
            for y in x:
                temp_sort.append(y)

        temp_sort.sort()
        indx = int(len(temp_sort) * percent / 100)
        low_temp = temp_sort[-indx]
        img = marker_image(self.thermal_values, self.visual, low_temp, color)
        return img

    def mix_visual_with_thermal_celsius(self, temp=None ):
        img= mix_image(self.thermal_values, self.visual, self.thermal ,temp)
        return img

    def mix_visual_with_thermal_precent(self,percent=100 ):
        temp_sort = []
        for x in self.thermal_values:
            for y in x:
                temp_sort.append(y)

        temp_sort.sort()
        indx = int(len(temp_sort) * percent / 100)
        low_temp = temp_sort[-indx]
        img = mix_image(self.thermal_values,  self.visual,self.thermal ,low_temp)
        return img

    def make_layer(self):
        p = ImageProcess(self.flir_path)
        pos = 0
        m = self.thermal_values.max()
        m= round(m,2)

        layer_arr = [(m - 2, "purple"), (m - 1.5, "blue"), (m - 1, "yellow"), (m-0.5, "white")]
        for x in layer_arr:
            temp, color = x

            i = p.filter_thermal_by_celsius(temp)
            edges = cv2.Canny(i, 100, 200)
            img = p.marker_visual_by_celsius(temp, color)
            cv2.putText(img, "{} > {}".format(color, temp), (10, 20 + pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color=color_bgr(color))
            hilight(img, edges)
            pos = pos + 30

        return p.show_visual()



p = ImageProcess(  r"C:\Users\zur99\Pictures\flir original\b.JPG")
a = []
for t in range(340, 400, 1):
    t =t/10
    a.append(p.mix_visual_with_thermal_celsius(t))
    p.reset()



with open("test.text", "wb") as fp:
    pickle.dump(a, fp)







