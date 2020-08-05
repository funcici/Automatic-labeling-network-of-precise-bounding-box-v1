# Automatic-labeling-network-of-precise-bounding-box-v1


用于减少工作量的自动标注网络，目前适用于汽车，需要手动标注中心点，并返回目标的最小外接矩形。本项目仅空天信息研究院二部使用

环境，python3.5+ ，pytorch ，cuda9.0+ ，cv2 等


仅仅使用

第一步，你需要使用标注软件中的点标注功能对车辆进行标注（你仅仅需要标注出车辆的近似中心点）

第二步，你需要将vif生成对应的csv文件.csv文件格式应和./file/ff.csv格式一致（如果想验证可以使用tif2csv.py文件，用之前标注好的斜框vif文件生成只有中心点的csv文件进行验证）

第三步，你需要下载我预训练好的模型放在合适位置，地址链接：https://pan.baidu.com/s/1I-aCNbT2L4h8itwBoSideg   提取码：9eox

第四步，你需要修改run.py中的各种（图片，csv文件，训练好的模型等）路径并且把模式改为application。运行run.py将生成结果图以及将中心点标注转化为斜框标注的new.vif存在file文件中


想要训练

下载预训练的resnet50，地址链接：https://pan.baidu.com/s/1dWSBxIdfIrJFaUPxZaMLQQ 提取码：8fdj

仅仅需要修改run.py中的模式 和datasetz文件中路径即可



训练数据

你可以使用之前标注过的斜框vif文件和图像，通过简单修改generateCsv.py和make_label.py来生成训练数据。



本模型利用mask的最小外接矩形生成box，在训练时使网络只关注中心目标，学习到位置偏见。生成的box质量远远优于简单回归的box。本项目中使用的模型为之前我标注的大图生成数据训练的，使用1050tiGPU，训练了100个epoch，由于没有细致调参和使用过多tricks，如果再次训练效果肯定会更好。

如果配合中心点检测网络使用将更加的智能。

