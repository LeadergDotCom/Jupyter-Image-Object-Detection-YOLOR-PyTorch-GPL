import os
"""
text_folder = "inference\\output\\"
image_folder = "inference\\output\\"
for text_file in os.listdir(text_folder):
    #print(text_file)
    if os.path.splitext(text_file)[1] == ".png":
        filename = os.path.splitext(text_file)[0] + ".txt"
        temp = os.path.join(image_folder, filename)
        if not os.path.exists(temp):
            print("del %s" %(os.path.join(text_folder, text_file)))
            #os.system("del %s" %(os.path.join(text_folder, text_file)))
os.system("pause")
"""

import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *
import threading


global model
global i 
imgsz = 512
dataset1 = LoadImages("data/Cu_dust-20180331-1_6_0_0.png", img_size=imgsz, auto_size=64)
dataset2 = LoadImages("data/Cu_dust-20180331-1_6_0_90.png", img_size=imgsz, auto_size=64)

def test():
    global model
    global i 
    print("test function")
    if i == 0:
        dataset = dataset1
    else:
        dataset = dataset2
    thread = threading.current_thread()
    player_name = thread.getName()
    print(player_name)
    img = torch.zeros((1, 3, imgsz, imgsz), device=device) 
    for path, img, im0s, vid_cap in dataset:
        
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        pred = model(img, augment=False)[0]
    print(pred)
    
def read_image():
    
    return  dataset1, dataset2
i = 0

cfg = "../data/yolor_csp_x.cfg"
weights = "data/best.pt"
device = select_device("0")
model = Darknet(cfg, imgsz).cuda()
model.load_state_dict(torch.load(weights, map_location=device)['model'])

model.to(device).eval()

img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
_ = model(img) if device.type != 'cpu' else None  # run once
    

    
    
threads = []
for i in range(2):
    thread_name = 'player' + str(i)
    print(thread_name + ' ready~')

    threads.append(threading.Thread(target=test, name=thread_name))
    threads[i].start()

for i in range(2):
    threads[i].join()
    print("Done.")
os.system("pause")