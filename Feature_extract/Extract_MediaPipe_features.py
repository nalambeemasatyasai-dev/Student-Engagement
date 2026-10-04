import cv2
import mediapipe as mp
import itertools
import numpy as np
import os
from time import time
import pandas as pd
import matplotlib.pyplot as plt

import mediaPipeFeatureExtractor as fmp

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
wacv_data_dir = os.path.join(BASE_DIR, "WACV data")

#path to WACV image folders (Classes 0, 1, 2)
path1 = os.path.join(wacv_data_dir, "0")
path2 = os.path.join(wacv_data_dir, "1")
path3 = os.path.join(wacv_data_dir, "2")


df_OF0 = pd.read_csv(os.path.join(wacv_data_dir, "processedData0.csv"))
df_OF1 = pd.read_csv(os.path.join(wacv_data_dir, "processedData1.csv"))
df_OF2 = pd.read_csv(os.path.join(wacv_data_dir, "processedData2.csv"))

lm_dic0 = fmp.faceMesh_extract(path1,False)
lm_dic1 = fmp.faceMesh_extract(path2,False)
lm_dic2 = fmp.faceMesh_extract(path3,False)

df_merge0 = fmp.buildFeatureDataframe(lm_dic0,0,df_OF0)
df_merge1 = fmp.buildFeatureDataframe(lm_dic1,1,df_OF1)
df_merge2 = fmp.buildFeatureDataframe(lm_dic2,2,df_OF2)

df_merge0.to_csv(os.path.join(wacv_data_dir, "merged_data0.csv"))
df_merge1.to_csv(os.path.join(wacv_data_dir, "merged_data1.csv"))
df_merge2.to_csv(os.path.join(wacv_data_dir, "merged_data2.csv"))