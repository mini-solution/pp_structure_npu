import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image,ImageOps
from shapely.geometry import Polygon
import pyclipper
from pdf2image import convert_from_path
import yaml

# 区域类型标签
label_list = [
    "paragraph_title",# 段落标题 
    "image",#图像
    "text",#文本
    "number",#页码
    "abstract",#摘要
    "content",# 目录
    "figure_title",#图/表标题
    "formula",#公式
    "table", # 表格
    "reference",# 参考文献
    "doc_title", # 文档标题
    "footnote", # 脚注
    "header", # 页眉
    "algorithm",# 算法
    "footer",#页脚
    "seal",#印章
    "chart", #图表
    "formula_number",#公式编号
    "aside_text",#侧栏文本
    "reference_content",#参考文献内容
]

class Layout:
    def __init__(self,
            pdf_path,
            model_dir='models/PP-DocLayout_plus-L',
            debug=False,
            output='output',
            providers=['DmlExecutionProvider', 'CPUExecutionProvider']
        ):
        self.pdf_file = pdf_path
        self.model_path = os.path.join(model_dir, "inference.onnx")
        self.model_conf_path = os.path.join(model_dir, "inference.yml")
        self.debug = debug
        self.output = output
        self.providers = providers
        # 获取类别定义
        with open(self.model_conf_path, "r", encoding="utf-8") as f:
            model_conf = yaml.safe_load(f)
            self.label_list =  model_conf["label_list"]
    # 图片前处理
    def pre_process(self,img, target_size=(800, 800)):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, target_size).astype(np.float32) / 255.0
        img_resized = img_resized.transpose(2, 0, 1)  # HWC -> CHW
        input_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32)  # NCHW
        return input_tensor
    # 解析版面分析的结果，
    # 并映射回原图尺寸
    def parse_output(self,outputs, orig_size, input_size=(800,800), score_thresh=0.3):
        orig_w, orig_h = orig_size
        model_w, model_h = input_size
        scale_x = orig_w / model_w
        scale_y = orig_h / model_h

        results = []
        for row in outputs:
            class_id = int(row[0])
            score = float(row[1])
            if score < score_thresh:
                continue
            x1, y1, x2, y2 = map(float, row[2:6])
            # 映射回原图
            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)
            results.append({
                "class_id": class_id,
                "label": self.label_list[class_id] if class_id < len(self.label_list) else f"cls_{class_id}",
                "score": score,
                "bbox": [x1, y1, x2, y2]
            })
        return results
    # 区域可视化标注
    def draw(self,image, results):
        img = image.copy()
        for obj in results:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            label = f'{obj["label"]} {obj["score"]:.2f}'
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(0, y1 - 5)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return img
    # 图片分割
    def crop(self,page,image,results,debug=False):
        cropped_results = []
        img = image.copy()
        for i,obj in enumerate(results):
            save_dir = f"{self.output}/crops/page"
            x1, y1, x2, y2 = map(int, obj["bbox"])
            cropped_array = img[y1:y2+5, x1:x2]
            # 创建文件夹
            os.makedirs(f"{save_dir}{page}", exist_ok=True) 
            # 转换为 PIL Image
            cropped_image = Image.fromarray(cropped_array)
            # cropped_image = ImageOps.expand(cropped_image, 
            #                               border=(0, 40, 0, 40), 
            #                               fill='white')  # 白色填充
            if debug:
                cropped_image.save(f"{save_dir}{page}/image_{i}_{obj['label']}_{obj['score']}.jpg", quality=95)
            cropped_results.append({
                "page": page,
                "label": obj['label'],
                "score": obj['score'],
                "image": cropped_image
            })
        return cropped_results 
    # 结果预测
    def predict(self):
        # pdf转为图片
        images = convert_from_path(self.pdf_file, dpi=300)
        session = ort.InferenceSession(self.model_path, providers=self.providers)
        layout_results = []
        for page, img in enumerate(images):
            img = np.array(img)
            src_h, src_w = img.shape[:2]
            # 图片尺寸缩放到640*640
            input_tensor = self.pre_process(img)
            outputs =  session.run(
                None,
                {
                    "image": input_tensor,
                    "im_shape": np.array([[src_h, src_w]], dtype=np.float32),  # 原图尺寸 (h, w)
                    "scale_factor": np.array([[src_h/800, src_w/800]], dtype=np.float32)
                }
            )
            # 解析推理结果
            results = self.parse_output(outputs[0], orig_size=(src_w, src_h), input_size=(800,800), score_thresh=0.5)
            results = sorted(results, key=lambda b: (b['bbox'][1]//10, b['bbox'][0]))
            # 打印解析结果
            # 并标注版面检测的结果
            if self.debug:
                result_img = self.draw(img, results)
                cv2.imwrite(f"layout_result_page_{page+1}.jpg", result_img)
            # 区域分割
            cropped_results = self.crop(page,img,results,self.debug)
            for result in cropped_results:
                layout_results.append(result)
        return layout_results