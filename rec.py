import os
import cv2
import numpy as np
import onnxruntime as ort
from numpy import ndarray
from PIL import Image
import yaml

class Rec:
    def __init__(
        self,
        image:ndarray,
        model_dir="models/PP-OCRv5_server_rec",
        providers=['DmlExecutionProvider', 'CPUExecutionProvider'],
        debug=False
    ):
        self.image = image 
        self.model_path = os.path.join(model_dir, "inference.onnx")
        self.providers = providers
        self.debug = debug
        self.model_conf_path = os.path.join(model_dir, "inference.yml")
        # 获取类别定义
        with open(self.model_conf_path, "r", encoding="utf-8") as f:
            model_conf = yaml.safe_load(f)
            self.character_dict =  [' '] + model_conf["PostProcess"]["character_dict"]
    #     return img
    def resize_rec_img(self,img, image_shape=(3, 48, 3200)):
        imgC, imgH, imgW = image_shape
        if img.shape[1] <= 160:
            imgW = 160
        elif img.shape[1] <= 320:
            imgW = 320
        h, w = img.shape[0:2]
        # 根据高度缩放，保持长宽比
        ratio = w / float(h)
        new_w = int(np.ceil(imgH * ratio))
        if new_w > imgW:
            new_w = imgW
        
        resized = cv2.resize(img, (new_w, imgH))

        # 使用127填充背景（不是0）
        padded = np.ones((imgH, imgW, 3), dtype=np.uint8) * 127
        padded[:, 0:new_w, :] = resized

        # 归一化
        img = padded.astype("float32") / 255
        img = (img - 0.5) / 0.5
        img = img.transpose((2, 0, 1))  # CHW
        img = np.expand_dims(img, axis=0)
        return img

    # 加载字典
    def load_dict(self,dict_path):
        with open(dict_path,'r',encoding='utf-8') as f:
            return [line.strip() for line in f]

    # 将概率数据转为文字
    def ctc_decode(self,preds, char_list, blank_idx='first'):
        if blank_idx == 'last':
            blank = preds.shape[2]-1
        else:
            blank = 0
        pred_indices = np.argmax(preds, axis=2)[0]
        prev_idx = -1
        text = ''
        for idx in pred_indices:
            if idx != prev_idx and idx != blank and idx < len(char_list):
                text += char_list[idx]
            prev_idx = idx
        return text
    # 执行预测
    def predict(self):
        # 加载字典
        # char_list = [' '] + self.load_dict('ppocrv5_dict.txt')
        # 加载模型
        session = ort.InferenceSession(self.model_path)
        rec_input_name = session.get_inputs()[0].name
        # 尺寸缩放
        rec_input = self.resize_rec_img(self.image)
        # 推理
        preds, = session.run(None, {rec_input_name: rec_input})
        
        if preds.shape[1] > preds.shape[2]:
            preds = np.transpose(preds, (0,2,1))

        text = self.ctc_decode(preds, self.character_dict, blank_idx='first')
        return text