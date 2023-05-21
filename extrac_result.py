import io
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import uuid
import json
import base64
import imageio
import cv2

import torch
import numpy as np
from PIL import Image
import streamlit as st
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from api import infer as inf
from backend.backend_utils import NpEncoder

import configs as cf
from backend.kie.kie_utils import (
    load_gate_gcn_net,
    run_predict,
    vis_kie_pred,
    postprocess_scores,
    postprocess_write_info,
)
from backend.backend_utils import create_merge_cells, get_request_api

def load_model():
    gcn_net = load_gate_gcn_net(cf.device, cf.kie_weight_path)
    config = Cfg.load_config_from_name("vgg_seq2seq")
    config["cnn"]["pretrained"] = False
    config["device"] = cf.device
    config["predictor"]["beamsearch"] = False
    detector = Predictor(config)

    return gcn_net, detector


gcn_net, detector = load_model()


def infer(img_fp, save_dir, random_id = None):

    img_fp["random_id"] = random_id
    api_random_id = img_fp["random_id"]
    cells = img_fp["cells"]

    imgdata = base64.b64decode(img_fp["image"])
    pil_img = Image.open(io.BytesIO(imgdata))
    img = np.array(pil_img)

    group_ids = np.array([i["group_id"] for i in cells])
    # merge adjacent text-boxes
    merged_cells = create_merge_cells(
        detector, img, cells, group_ids, merge_text=cf.merge_text
    )
    batch_scores, boxes,text_all = run_predict(gcn_net, merged_cells, device=cf.device)

    # 2 options: get max score or filter categories by threshold
    values, preds = postprocess_scores(
        batch_scores, score_ths=cf.score_ths, get_max=cf.get_max
    )
    kie_info = postprocess_write_info(merged_cells, preds)

    # visualize prediction
    save_path = os.path.join(save_dir, "{}.jpg".format(api_random_id))
    vis_img = vis_kie_pred(img, preds, values, boxes, save_path)

    return img,vis_img, kie_info,text_all

def total_number_strings(text_all):
    total = 0
    for str__ in text_all:
        num = len(str__)
        total += num
    return total


if __name__ == '__main__':
    img_path="images\\mcocr_public_145013bchvc.jpg"
    random_id = str(uuid.uuid4())
    image = np.array(Image.open(img_path))
    img_info = inf(image, random_id)
    save_dir=cf.result_img_dir

    img,vis_img,kie_info,text_all=infer(img_info,save_dir)
    total_str = total_number_strings(text_all)
    uni_show=np.concatenate((img,vis_img),axis=1)

    print("So cau: ",len(text_all)) 
    print("Tong so luong ki tu:", total_str)
    print(text_all)
    # cv2.imshow('result',uni_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()






    

