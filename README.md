# Automatic-labeling-network-of-precise-bounding-box-v1


用于减少工作量的自动标注网络，目前适用于汽车，需要手动标注中心点，并返回目标的最小外接矩形。本项目仅空天信息研究院二部使用

环境，python3.5+ ，pytorch ，cuda9.0+ ，cv2 等

仅仅使用

第一步，你需要使用标注软件中的点标注功能对车辆进行标注（你仅仅需要标注出车辆的近似中心点）

第二步，你需要将vif生成对应的csv文件.csv文件格式应和./file/ff.csv格式一致（如果想验证可以使用tif2csv.py文件，用之前标注好的斜框vif文件生成只有中心点的csv文件进行验证）

第三步，你需要修改run.py中的各种（图片，csv文件，训练好的模型等）路径并且把模式改为application。运行run.py将生成结果图以及将中心点标注转化为斜框标注的new.vif存在file文件中


想要训练

仅仅需要修改run.py中的模式 和datasetz文件中路径即可

训练数据

你可以使用之前标注过的斜框vif文件和图像，通过文件



