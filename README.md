# Automatic-labeling-network-of-precise-bounding-box-v1


用于减少工作量的自动标注网络，目前适用于汽车，需要手动标注中心点，并返回目标的最小外接矩形。本项目仅空天信息研究院二部使用

环境，python3.5+ ，pytorch ，cuda9.0+ ，cv2 等


仅仅使用

第一步，你需要使用标注软件中的点标注功能对车辆进行标注（你仅仅需要标注出车辆的近似中心点）
v
第二步，你需要将第一步中的vif生成对应的csv文件（使用vifdian2csv.py文件）. csv文件格式应和./file/ff.csv格式一致（如果想验证可以使用vif2csv.py，将之前标注好的斜框vif文件生成只有中心点的csv文件进行验证）

第三步，你需要下载我预训练好的模型放在合适位置，地址链接：https://pan.baidu.com/s/1I-aCNbT2L4h8itwBoSideg   提取码：9eox

第四步，你需要修改run.py中的各种（图片，csv文件，训练好的模型等）路径并且把模式设置为application。
修改longitudeoffset和latitudeoffset，这两个值分别为PZ中-层管理器-影像层-影像信息-（最小经度和最大维度）
运行run.py将会生成结果图以及中心点标注转化为斜框标注的new.vif保存在file文件中


想要训练

下载预训练的resnet50，地址链接：https://pan.baidu.com/s/1dWSBxIdfIrJFaUPxZaMLQQ 提取码：8fdj

仅仅需要修改run.py中的模式 和dataset文件中路径即可



训练数据

你可以利用之前标注好的斜框vif文件和图像，通过使用简单修改后的generateCsv.py和make_label.py来生成训练数据。



本模型利用中心目标mask的最小外接矩形生成bounding box，在训练时使网络只关注中心目标，学习到位置偏见。生成的bounding box质量远远优于简单回归的bounding box。本项目中使用的模型为之前我标注的大图生成数据训练的，使用1050tiGPU，训练了100个epoch，的模型。 由于没有细致调参和使用过多tricks，如果再次训练效果肯定会更好。

如果配合中心点检测网络使用将更加的智能。本项目会继续修改使使用更加方便


如果对你有帮助，点个星星吧。
