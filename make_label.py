import os
from PIL import Image
import pandas as pd
import csv
import cv2
point = pd.read_csv('../file/ff9.csv')  # 读取数据
image_dir = "../file/ff9.tif"
image_new_dir = "../image1/"
label_dir = "../label/"
edge_dir = "../edge/"
#f = open('../cut_image/image.txt','a')
csvFile = open('../cut_image/image.lst', "w")
writer = csv.writer(csvFile)                  #创建写的对象
w = 32
h = 32
#image = Image.open(image_dir)
image = cv2.imread(image_dir)
image=cv2.copyMakeBorder(image,h,h,w,w,cv2.BORDER_CONSTANT,value=(0,0,0))
ii = 0
s = image.shape
print(s)

ii = 0
class isPointInQuadrangle(object):

    def __int__(self):
        self.__isInQuadrangleFlag = False

    def cross_product(self, xp, yp, x1, y1, x2, y2):
        return (x2 - x1) * (yp - y1)-(y2 - y1) * (xp - x1)

    def compute_para(self, xp, yp, xa, ya, xb, yb, xc, yc, xd, yd):
        cross_product_ab = isPointInQuadrangle().cross_product(xp, yp, xa, ya, xb, yb)
        cross_product_bc = isPointInQuadrangle().cross_product(xp, yp, xb, yb, xc, yc)
        cross_product_cd = isPointInQuadrangle().cross_product(xp, yp, xc, yc, xd, yd)
        cross_product_da = isPointInQuadrangle().cross_product(xp, yp, xd, yd, xa, ya)
        return cross_product_ab,cross_product_bc,cross_product_cd,cross_product_da

    def is_in_rect(self, aa, bb, cc, dd):
        if (aa > 0 and bb > 0 and cc > 0 and dd > 0) or (aa < 0 and bb < 0 and cc < 0 and dd < 0):
            print("This point is in the Quadrangle.")
            self.__isInQuadrangleFlag= True
        else:
            print("This point is not in the Quadrangle.")
            self.__isInQuadrangleFlag = False

        return self.__isInQuadrangleFlag

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
for i in point["name"]:
    print(i)
    xs = int(point['center_point_x'][ii])
    ys = int(point['center_point_y'][ii])
    xe =int(point['center_point_x'][ii])+(2*w)
    ye = int(point['center_point_y'][ii])+(2*h)
    print(xs,ys,xe,ye)
    image_new = image[ys:ye,  xs:xe]
    image_new2 = image_new
    image_new2 = cv2.cvtColor(image_new2, cv2.COLOR_BGR2GRAY)

    image_new = cv2.resize(image_new, (128, 128), interpolation=cv2.INTER_LINEAR)
    image_new2 = cv2.resize(image_new2, (128, 128), interpolation=cv2.INTER_LINEAR)

    x1 = (int(point['x1'][ii]) - int(point['center_point_x'][ii]) + w)*2
    y1 = (int(point['y1'][ii]) - int(point['center_point_y'][ii]) + h)*2
    x2 = (int(point['x2'][ii]) - int(point['center_point_x'][ii]) + w)*2
    y2 = (int(point['y2'][ii]) - int(point['center_point_y'][ii]) + h)*2
    x3 = (int(point['x3'][ii]) - int(point['center_point_x'][ii]) + w)*2
    y3 = (int(point['y3'][ii]) - int(point['center_point_y'][ii]) + h)*2
    x4 = (int(point['x4'][ii]) - int(point['center_point_x'][ii]) + w)*2
    y4 = (int(point['y4'][ii]) - int(point['center_point_y'][ii]) + h)*2
    print(x1,y1,x2, y2, x3, y3, x4, y4)
    for iii in range(image_new2.shape[0]) :
        for iiii in range (image_new2.shape[1]) :
            print(iii,iiii)
            aa, bb, cc, dd = isPointInQuadrangle().compute_para(iiii, iii, x1, y1, x2, y2, x3, y3, x4, y4)
            if (isPointInQuadrangle().is_in_rect(aa, bb, cc, dd)):
                image_new2[iii][iiii]=255
            else:
                image_new2[iii][iiii]=0



    # 腐蚀图像
    image_new2 = cv2.erode(image_new2, kernel)
    image_new2 = cv2.erode(image_new2, kernel)


    c_canny_img = cv2.Canny(image_new2, 50, 150)
    ii += 1
    cv2.imwrite(image_new_dir +'9_'+ str(ii) +'.tif', image_new)
    cv2.imwrite(label_dir +'9_'+ i + '.tif', image_new2)
    cv2.imwrite(edge_dir + '9_' + i + '.tif', c_canny_img)



    #writer.writerow([i +'.tif'])
