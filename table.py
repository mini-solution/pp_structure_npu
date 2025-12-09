import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import yaml
from det import Det


# 表格结构检查
class TableStructRec:
    def __init__(
        self,
        image: Image,
        model_dir="models/SLANet",
        debug=False,
        output="output",
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    ):
        self.image = image
        self.model_path = os.path.join(model_dir, "inference.onnx")
        self.model_conf_path = os.path.join(model_dir, "inference.yml")
        self.debug = debug
        self.output = output
        self.providers = providers
        # 获取类别定义
        with open(self.model_conf_path, "r", encoding="utf-8") as f:
            model_conf = yaml.safe_load(f)
            self.character_dict =  model_conf["PostProcess"]["character_dict"]
            if "<td></td>" not in self.character_dict:
                self.character_dict.append("<td></td>")
            if "<td>" in self.character_dict:
                self.character_dict.remove("<td>")
            self.character_dict = ["sos"] + self.character_dict + ["eos"]
    # 图片前处理
    def pre_process(self, img, target_size=(488, 488)):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, target_size).astype(np.float32) / 255.0
        img_resized = img_resized.transpose(2, 0, 1)  # HWC -> CHW
        input_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32)  # NCHW
        return input_tensor

    # 解析结果 获取类别和每个单元格的属性
    def parse_output(self, output):
        model_conf = []
        parse_res = []
        for index, probs in enumerate(output[1][0]):
            cls_id = np.argmax(probs)  # 找到概率最大的类别索引
            if cls_id > len(self.character_dict)-1:
                continue
            label = self.character_dict[cls_id]
            if label == "eos":
                break
            parse_res.append(self.character_dict[cls_id])
        return parse_res

    # 结果预测
    def predict(self):
        img = np.array(self.image)
        input_tensor = self.pre_process(img)
        session = ort.InferenceSession(self.model_path, providers=self.providers)
        output = session.run(None, {session.get_inputs()[0].name: input_tensor})
        np.set_printoptions(threshold=np.inf)
        return self.parse_output(output)

# 表格分类(有线表格/无线表格)
class TableCls:
    def __init__(
        self,
        image: Image,
        model_dir="models/PP-LCNet_x1_0_table_cls",
        debug=False,
        output="output",
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    ):
        self.image = image
        self.model_path = os.path.join(model_dir, "inference.onnx")
        self.model_conf_path = os.path.join(model_dir, "inference.yml")
        self.debug = debug
        self.output = output
        self.providers = providers
        with open(self.model_conf_path, "r", encoding="utf-8") as f:
            model_conf = yaml.safe_load(f)
            self.label_list = model_conf["PostProcess"]["Topk"]["label_list"]

    # 图片预处理
    def pre_process(self, img, target_size=(224, 224)):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, target_size).astype(np.float32) / 255.0
        img_resized = img_resized.transpose(2, 0, 1)  # HWC -> CHW
        input_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32)  # NCHW
        return input_tensor

    # 解析结果类别
    def parse_output(self, output):
        cls_id = np.argmax(output[0][0])  # 找到概率最大的类别索引
        return self.label_list[cls_id]

    def predict(self):
        img = np.array(self.image)
        input_tensor = self.pre_process(img)
        session = ort.InferenceSession(self.model_path, providers=self.providers)
        output = session.run(None, {session.get_inputs()[0].name: input_tensor})
        return self.parse_output(output)


# 有线单元格检测
class TableDet:
    def __init__(
        self,
        image: Image,
        model_dir="models/RT-DETR-L_wired_table_cell_det",
        debug=False,
        output="output/table_cell_wired",
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    ):
        self.image = image
        self.model_path = os.path.join(model_dir, "inference.onnx")
        self.model_conf_path = os.path.join(model_dir, "inference.yml")
        self.debug = debug
        self.output = output
        self.providers = providers
    # 图片预处理
    def pre_process(self,img, target_size=(640, 640)):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, target_size).astype(np.float32) / 255.0
        img_resized = img_resized.transpose(2, 0, 1)  # HWC -> CHW
        input_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32)  # NCHW
        return input_tensor
    # 并映射回原图尺寸
    def parse_output(self,outputs, orig_size, input_size=(640,640), score_thresh=0.5):
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
                "score": score,
                "bbox": [x1, y1, x2, y2]
            })
        return results
    # 区域可视化标注
    def draw(self,image, results):
        img = image.copy()
        for obj in results:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return img
    # 图片分割
    def crop(self,image,results,debug=False):
        cropped_results = []
        img = image.copy()
        for i,obj in enumerate(results):
            save_dir = f"{self.output}/crops"
            x1, y1, x2, y2 = map(int, obj["bbox"])
            cropped_array = img[y1:y2+5, x1:x2]
            # 创建文件夹
            os.makedirs(f"{save_dir}", exist_ok=True) 
            # 转换为 PIL Image
            cropped_image = Image.fromarray(cropped_array)
            # cropped_image = ImageOps.expand(cropped_image, 
            #                               border=(0, 40, 0, 40), 
            #                               fill='white')  # 白色填充
            if debug:
                cropped_image.save(f"{save_dir}/image_{i}_{obj['score']}.jpg", quality=95)
            cropped_results.append({
                "score": obj['score'],
                "image": cropped_image
            })
        return cropped_results 
    def predict(self):
        img = np.array(self.image)
        src_h, src_w = img.shape[:2]
        # input_tensor,scale,(new_w, new_h) = self.pre_process(img)
        input_tensor = self.pre_process(img)
        session = ort.InferenceSession(self.model_path, providers=self.providers)
        output = session.run(
            None, 
            {
                "im_shape": np.array([[src_h, src_w]], dtype=np.float32),  # 原图尺寸 (h, w)
                "scale_factor": np.array([[src_h/640, src_w/640]], dtype=np.float32),
                "image": input_tensor
            }
        )
        # 解析推理结果
        results = self.parse_output(output[0], orig_size=(src_w, src_h), input_size=(640,640), score_thresh=0.5)
        results = sorted(results, key=lambda b: (b['bbox'][1]//10, b['bbox'][0]))
        # 打印解析结果
        # 并标注版面检测的结果
        if self.debug:
            print(results)
            result_img = self.draw(img, results)
            cv2.imwrite(f"{self.output}/table_cell_result.jpg", result_img)
        # 区域分割
        table_cell = []
        cropped_results = self.crop(img,results,self.debug)
        for result in cropped_results:
            table_cell.append(result)
        return table_cell
# 表格识别
class TableRec:
    def __init__(
        self,
        image: Image,
        debug=False,
        output="output",
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    ):
        self.image = image
        self.debug = debug
        self.output = output
        self.providers = providers
    def predict(self):
        structKVs = []
        # 获取表格结构
        tableStructRec = TableStructRec(image=self.image)
        structLabels = tableStructRec.predict()
        # 获取表格类别
        tableCls = TableCls(image=self.image)
        tableCls = tableCls.predict()
        cell_res = []
        if tableCls == "wired_table":
            tableWiredDet = TableDet(image=self.image)
            cell_res = tableWiredDet.predict()
        else:
            tableWirelessDet = TableDet(image=self.image,model_dir="models/RT-DETR-L_wireless_table_cell_det")
            cell_res = tableWirelessDet.predict()
        tdIndex = 0
        for label in structLabels:
            if label in ["</td>","<td></td>"] and tdIndex <= len(cell_res)-1:
                cell = cell_res[tdIndex]
                tdIndex = tdIndex + 1
                det = Det(image=cell["image"])
                contents = det.predict()
                print(contents)
                if label == "</td>":
                    label = "".join(contents)+"</td>"
                if label == "<td></td>":
                    label = "<td>"+"".join(contents)+"</td>"
            structKVs.append(label)
        return "<table>"+"".join(structKVs)+"</table>"