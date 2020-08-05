import pandas as pd
import numpy as np
import re
infile = open("../file/ff10.vif", 'r', encoding='utf-8')
allPos = [] #保存名称对应的坐标
lastAns = [] #保存原始名称
lastname = [] #保存原始名称
point = []
cx = []
cy = []
axis_id = 0
xy2 =[]
#x_0 =14456076.5374056
#y_0 =3866872.82391877
#det_x =0.298582141738734
#det_y = 0.298582141739189
for i in infile:
    #if "VectorItem" in i:
        # print(i[35:60].split("\"")[1])
     #   if "矩形" in i:
     #       lastAns.append([i[35:60].split("\"")[1], "",i[35:60].split("\"")[1].replace("矩形","")+"_Small Car"])
     #   else:
     #       lastAns.append([i[35:60].split("\"")[1],"",i[35:60].split("\"")[1]])
    if 'longitudeoffset' in i:
        pattern = re.compile(r'\"(.*?)\"')
        ans = pattern.findall(i)
        x_0 = np.float64(ans[0])
        y_0 = np.float64(ans[1])
        print(x_0,y_0)
        det_x = np.float64(ans[2])
        det_y = np.float64(ans[3])
        print(det_x,det_y)
    if "<GeoShape>" in i:
        x = []
        y = []

    if "GeoShapePoint" in i:
        pattern = re.compile(r'\"(.*?)\"')
        ans = pattern.findall(i)
        axis_id += 1
        x.append(np.float64(ans[0]))
        y.append(np.float64(ans[1]))
    xy = np.zeros((4, 2), dtype=np.int32)
    if axis_id == 4:
        x = np.array(x)
        y = np.array(y)
        axis_id = 0
        x = np.abs(x - x_0) / det_x
        y = np.abs(y - y_0) / det_y
        cx.append(int(np.sum(x)/4))
        cy.append(int(np.sum(y)/4))

        for k in range(len(x)):
            xy[k][0] = np.int(x[k])
            xy[k][1] = np.int(y[k])
        xy2 = xy

        print(xy)

        poly_xy = xy.reshape(-1, 1, 2)
        allPos.append(
            {
                "poly_xy":poly_xy,
                "xy":xy
            }
        )

    if "VectorItem" in i:
            # print(i[35:60].split("\"")[1])
        if "矩形" in i:
            lastAns.append([i[35:60].split("\"")[1], "", i[35:60].split("\"")[1].replace("矩形", "") + "_Small Car"])
        else:
            lastAns.append([i[35:60].split("\"")[1], "", i[35:60].split("\"")[1]])
        lastname.append(i[35:60].split("\"")[1])
        print(i[35:60].split("\"")[1])
np.save('../file/allPos.npy', allPos, allow_pickle=True)
data = pd.DataFrame(lastAns, index=None, columns=["source name", "angle","last name"])


ii = 0
for name in  lastname:
    point.append([name,allPos[ii]['xy'][0][0],allPos[ii]['xy'][0][1],allPos[ii]['xy'][1][0],allPos[ii]['xy'][1][1],
    allPos[ii]['xy'][2][0],allPos[ii]['xy'][2][1],allPos[ii]['xy'][3][0],allPos[ii]['xy'][3][1],cx[ii],cy[ii]])
    ii += 1

data2 = pd.DataFrame(point, index=None, columns=["name","x1","y1","x2","y2","x3","y3","x4","y4","center_point_x","center_point_y"])
data.to_csv("../file/info.csv", index=None)
data2.to_csv("../file/ff10.csv", index=None)
print("Saved finished.")
print(allPos[0]['xy'][1][1])
