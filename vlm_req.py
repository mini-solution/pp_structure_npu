import cv2
import numpy as np
from PIL import Image
import base64
import requests
import io
class VlmReq:
    def __init__(
            self,
            image:Image,
            vlm_api="http://localhost:1234/v1/chat/completions",
            vlm_model="google/gemma-3-12b"
        ):
        self.image = image
        self.api = vlm_api
        self.model = vlm_model
    def pil_to_base64(self,img:Image, fmt: str = "PNG") -> str:
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        buffered = io.BytesIO()
        img.save(buffered, format=fmt)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    # 读取文件并转为base64格式
    def file_to_base64(path: str) -> str:
        # 以二进制模式读取文件
        with open(path, "rb") as f:
            file_data = f.read()
        # 转为base64并解码为utf-8字符串
        return base64.b64encode(file_data).decode("utf-8")
    def request(self):
        # 将输入图片转为base64格式
        base64_str = self.pil_to_base64(self.image)
        # print("base64_str",base64_str)
        # 调用ollama接口
        # 关闭流式输出
        payload = {
            "model": self.model, # 模型名称
            "stream": False, # 关闭流式输出
            "messages":[{
                "role":"user",
                "content":[
                    {
                        "type":"text",
                        "text":"获取图片的摘要信息"
                    },
                    {
                        "type": "image_url",
                        "image_url":{
                            "url": "data:image/jpeg;base64,"+base64_str
                        },
                    }
                ]
            }],
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api, json=payload, headers=headers)
        # 打印识别的文字
        return response.json()["choices"][0]["message"]["content"]