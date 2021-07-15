import numpy as np
import codecs
import cv2
import os
import json 

def read_txt(label_path):
    file = open(label_path,'r',encoding='utf-8')
#     res = file.read()  #读文件
#     json_result = json.loads(res)
    json_result = json.load(file)
    if json_result is None:
        return None
#     print(type(json_result))
#     print(len(json_result))
#     print(label_path)
    result_list = list()
    for i in range(len(json_result)): #取出list中的所有dict
        result =json_result[i]
        name = result['name']
        source = result['source']
        coords = result['coordinates']  #coords又是一个list，长度为4,有效坐标在第一个和第三个中
        xtl, ytl = float(coords[0]['axisX']), float(coords[0]['axisY'])
        xbr, ybr = float(coords[2]['axisX']), float(coords[2]['axisY'])       
        result_list.append(dict(box_coord=[round(xtl),round(ytl),round(xbr),round(ybr)], box_name=name))
       
    return result_list
    
    
def covert_xml(label,xml_path, img_name, img_path):
    # 获得图片信息
    img = cv2.imread(img_path)
    height, width, depth = img.shape

#     x_min,y_min,x_max,y_max = label
 
    xml = codecs.open(xml_path, 'w', encoding='utf-8')
    xml.write("<?xml version='1.0' encoding='utf-8'?>\n")
    xml.write('<annotation>\n')
    xml.write('\t<folder>' + 'JPEGImages' + '</folder>\n')
    xml.write('\t<filename>' + img_name + '</filename>\n')
#     xml.write('\t<source>\n')
#     xml.write('\t\t<database>The VOC 2007 Database</database>\n')
#     xml.write('\t\t<annotation>Pascal VOC2007</annotation>\n')
#     xml.write('\t\t<image>flickr</image>\n')
#     xml.write('\t\t<flickrid>NULL</flickrid>\n')
#     xml.write('\t</source>\n')
#     xml.write('\t<owner>\n')
#     xml.write('\t\t<flickrid>NULL</flickrid>\n')
#     xml.write('\t\t<name>faster</name>\n')
#     xml.write('\t</owner>\n')
    xml.write('\t<size>\n')
    xml.write('\t\t<width>' + str(width) + '</width>\n')
    xml.write('\t\t<height>' + str(height) + '</height>\n')
    xml.write('\t\t<depth>' + str(depth) + '</depth>\n')
    xml.write('\t</size>\n')
#     xml.write('\t\t<segmented>0</segmented>\n')
    for i in range(len(label)):
        infos = label[i]
#         import pdb; pdb.set_trace()
        box_name = infos.get('box_name')
        if box_name == '车辆整体' or box_name == '卡车客车等车头':
            box_name = '车脸'
        coords = infos['box_coord']
        xml.write('\t<object>\n')
        xml.write('\t\t<name>' + box_name + '</name>\n')
        xml.write('\t\t<difficult>0</difficult>\n')
        xml.write('\t\t<bndbox>\n')
        xml.write('\t\t\t<xmin>' + str(coords[0]) + '</xmin>\n')
        xml.write('\t\t\t<ymin>' + str(coords[1]) + '</ymin>\n')
        xml.write('\t\t\t<xmax>' + str(coords[2]) + '</xmax>\n')
        xml.write('\t\t\t<ymax>' + str(coords[3]) + '</ymax>\n')
        xml.write('\t\t</bndbox>\n')
        xml.write('\t</object>\n')
    
    xml.write('</annotation>')


    labels_file_path = "./gpfs/application/label/download/bfb4c990cc034b57b12e45ea5eb96d23/imageszh_zip/imageszh"
    imgs_file_path = "JPEGImages"
    xmls_file_path = "Annotations"
    if not os.path.exists(xmls_file_path):
        os.mkdir(xmls_file_path)
 
    labels_name = os.listdir(labels_file_path)
#     i=0
    for label_name in labels_name:
        label_path = os.path.join(labels_file_path, label_name)
        label = read_txt(label_path)
        if label is None:
            continue
        
        xml_name = label_name[:-4]+'.xml'
        xml_path = os.path.join(xmls_file_path, xml_name)
 
        img_name = label_name[:-4]+'.jpg'
        img_path = os.path.join(imgs_file_path, img_name)
        
#         print(img_path)
#         print(xml_path)
        
        covert_xml(label, xml_path, img_name, img_path)
#         i=i+1
#         if i > 10:
#             break


# 统计Annotations中的文件数，也就是整个训练集测试集的总图片张数,
# 分成两个数据集
xmls_file_path = "Annotations"
labels_name = os.listdir(xmls_file_path)
total_nmus = len(labels_name)
idx = np.arange(total_nmus)
np.random.shuffle(idx)
shuffled_labels_name = labels_name[idx,...]

f_train = open('train.txt','w')
f_test = open('test.txt','w')
    
for i, label_name in enumerate(shuffled_labels_name) :
     file_name = label_name[:-4]
     if i< total_nmus/10 :
        f_train.write('{}\n'.format(file_name))
     else:
        f_test.write('{}\n'.format(file_name))    