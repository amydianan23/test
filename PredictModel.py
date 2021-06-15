# predict
import os
import math
import cx_Oracle
import cv2
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import datetime
import winsound
import keras.backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import configparser
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
# In[相關初始值設定]
now = datetime.datetime.now() #現在時間
date=now + datetime.timedelta()
day=date.strftime('%Y%m%d') #完整日期_上線
dataformate=date.strftime("%Y-%m-%d %H:%M:%S") #上線版本
#day='20210331'
# In[root]
currentroot='D:/Camy/AI_project/CELL_ASM_SealWidth_SPC_N2/code' #絕對路徑，相對路徑os.getcwd()
os.chdir(currentroot)
#currentroot=os.getcwd()
down_img=currentroot+'/input/Downpic/'
input_img=currentroot+'/input/images/'
image_compare=currentroot+'/input/compare/'
output_img= currentroot+'/output/yoloimage/' #/image/
output_csv= currentroot+'/output/csv/' #/image/
#output_csv= currentroot+'/output/csv_data/'
#check_table=currentroot+'/CheckTable'
Models_path=currentroot+'/models/'
##Creat file
makedirs(down_img)
makedirs(input_img)
makedirs(image_compare)
makedirs(output_img)
makedirs(output_csv)
#makedirs(output_csv)
#makedirs(check_table)
# In[readConfig]
config = configparser.ConfigParser()
config.read('Config.ini',encoding='utf-8')#config檔案名稱
ModelName = list(config['Setting']['model_Name'].split(','))
ModelH5File = list(config['Setting']['model_h5Nmae'].split(','))
SheetModel_Modelname_dic={}
for d in range(len(ModelName)):
    SheetModel_Modelname_dic[ModelName[d]]=ModelH5File[d]
# In[Set Model some parameter]
net_h, net_w = 288, 288 # a multiple of 32, the smaller the faster
obj_thresh, nms_thresh = 0.6, 0.4   #num 0.75 0.4
#anchors = [0,0, 9,78, 10,96, 10,75, 13,85, 53,13, 70,15, 72,11, 85,13]  #models_200622
anchors = [20,72, 26,73, 29,74, 32,79, 32,69, 37,75, 41,77, 47,73, 49,78]   #asm
labels = ["seal width"] #LV1_02-26
# In[函式設定]
#影像前處理會用到的
def preCV(downroot,inputimgroot):
    for img_p in os.listdir(downroot):
        os.chdir(downroot)
        dataimage = cv2.imread(img_p,0)#灰階
        ##---1.均衡化+前處理---#
        equ_hist=cv2.equalizeHist(dataimage)
        final_image=modify_contrast_and_brightness(equ_hist,int(-50) ,int(0))
        os.chdir(inputimgroot)
        cv2.imwrite(img_p,final_image)
        ##合併檔案
        os.chdir(image_compare)
        compare_hist = np.hstack((dataimage, final_image)) # 原圖跟+final
        cv2.imwrite('Compare_'+img_p,compare_hist)

def modify_contrast_and_brightness(img,brightness,contrast):
    #contrast = - 減少對比度/(白黑都接近灰)+ 增加對比度
    B = brightness / 255.0
    c = contrast / 255.0 
    k = math.tan((45 + 44 * c) / 180 * math.pi)
    img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
    # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def image_path_jpg(image_path):
    image_paths = []
    if os.path.isdir(image_path):
        for inp_file in os.listdir(image_path):
            image_paths += [inp_file]
    else:
        image_paths += [image_path]
    
    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG', '.PNG','.bmp','.BMP'] and inp_file[-7:]!='MAP.jpg')]
    return image_paths

def Cau_Value(imgname,data,ratio,type):  
    if type==0:
        B=pd.DataFrame(data[1]) #   box_value[1]
        B.columns=['label','Xmin','Ymin','Xmax','Ymax','obj_percent']   
        B['Value']=B['Xmax']-B['Xmin'] #B['Ymax']-B['Ymin']
        sw_value=int(B['Value'].min()*float(ratio)) 
        B['SWValue']=sw_value
        B['ImageName']=imgname
        B=B[B['Value']==B['Value'].min()] #需修改，可能出現兩個都最小
    else:
        B=pd.DataFrame([imgname],columns=['ImageName'])
        B['label']='NAN'
        B['Xmin']='NAN'
        B['Ymin']='NAN'
        B['Xmax']='NAN'
        B['Ymax']='NAN'
        B['obj_percent']='NAN'
        B['Value']='NAN'
        B['SWValue']='NAN'
    return B

def datasave(saveroot,Finaldf,filename):
    Data_name=filename+'.csv' #檔名使用機台名稱
    his = list(filter(lambda x: x[0: len(Data_name)]==Data_name, os.listdir(saveroot)))
    os.chdir(saveroot)  
    if len(his)==1:
        Hisdata=pd.read_csv(filename+'.csv')
        #Hisdata['DATETIME']=pd.to_datetime(Hisdata['DATETIME'])
        Condata=pd.concat([Hisdata,Finaldf],axis=0,ignore_index=False)
        #Condata['DATETIME']=pd.to_datetime(Condata['DATETIME'])
        #Condata.sort_values(by='DATETIME',inplace=True)
        #Condata.drop_duplicates(['ImageName'],keep='first',inplace=True)
        Condata.to_csv(filename+'.csv',index=False)
    else:
        Finaldf.to_csv(filename+'.csv',index=False)
        
# In[ASM 主程式01_照片預測] 
#infor_model
os.chdir(Models_path)
infer_model=load_model('B116XTN4B_20210419.h5')
#infer_model=load_model(SheetModel_Modelname_dic[GET_KPCData(input_path_03_3[k])[1].split('_')[0]])
##---1.Preimage CV ---##
preCV(down_img,input_img) #照片先抓到down_img後，預處理照片儲存志input_img
#read img
for img_p in os.listdir(input_img):
    ##---2.yolo-model---##
    os.chdir(input_img)
    image = cv2.imread(img_p)#讀取一般bgr照片
    # predict the bounding boxes
    boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, anchors, obj_thresh, nms_thresh)[0]
    # draw bounding boxes on the image using labels
    draw_boxes(image, boxes, labels, obj_thresh)
    #從BBOX取出BOXE_SIZE
    box_value=draw_boxes(image, boxes, labels, obj_thresh)  
    if len(box_value[1])>0:
        #計算數值
        datasave(output_csv,Cau_Value(img_p,box_value,float(9.67),0),'ASM_sealwidth_'+day)
        #儲存照片
        output_imgname=output_img+img_p
        print(output_imgname)
        cv2.imwrite(output_imgname, np.uint8(image))
    else:
        datasave(output_csv,Cau_Value(img_p,box_value,float(9.67),1),'ASM_sealwidth_'+day)
os.chdir(currentroot)   


#Offline_合併檔案
#finsave=r'D:\Camy\AI_project\CELL_ASM_SealWidth_SPC_N2\ML5CN2_AsmSealWidth\TestPicture\B116XTN4B\Test'
#for i in os.listdir(down_img):
#    os.chdir(down_img)
#    a=cv2.imread(i)
#    os.chdir(output_img)
#    b=cv2.imread(i)
#    compare_hist = np.hstack((a, b)) # 原圖跟+final
#    os.chdir(finsave)
#    cv2.imwrite(i,compare_hist)
#
#
