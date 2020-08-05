
# 从txt中读取行
def load_from_txt(filename, encoding="utf-8"):
    f = open(filename,'r' ,encoding=encoding)
    contexts = f.readlines()
    return contexts

# 坐标转经纬度
def xy2geo(xy, pixel, init_geo):
    x,y = xy
    geo_x,geo_y = init_geo
    x = (x*pixel)+geo_x
    y = -(y*pixel)+geo_y
    return x,y

# csv item解析
def transCsvItem(idx, item):
    name = str(idx)+"_"+item[0][2:]
    factor = 2

    pos = []
    pos.append([item[1]/factor,item[2]/factor])
    pos.append([item[3]/factor,item[4]/factor])
    pos.append([item[5]/factor,item[6]/factor])
    pos.append([item[7]/factor,item[8]/factor])
    return name,pos