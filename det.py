import os
import cv2
import numpy as np
import onnxruntime as ort
from shapely.geometry import Polygon
import pyclipper
from PIL import Image
from rec import Rec
# 文字识别后处理
# 将概率数据转为每个文字区域的坐标
class DBPostProcess:
    def __init__(self, thresh=0.3, box_thresh=0.5, max_candidates=1000, unclip_ratio=2.0):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap*255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        scores = []
        for contour in contours[:self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape(-1, 2)
            if points.shape[0] < 3:  # 至少三点
                continue
            score = self.box_score_fast(pred, points)
            if score < self.box_thresh:
                continue
            box = self.unclip(points)
            if box is None or len(box) < 3:
                continue
            box = np.array(box)
            box[:,0] = np.clip(np.round(box[:,0] / width * dest_width), 0, dest_width)
            box[:,1] = np.clip(np.round(box[:,1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def box_score_fast(self, bitmap, box):
        h, w = bitmap.shape
        box = np.array(box)
        xmin = np.clip(np.floor(box[:,0].min()).astype(int), 0, w-1)
        xmax = np.clip(np.ceil(box[:,0].max()).astype(int), 0, w-1)
        ymin = np.clip(np.floor(box[:,1].min()).astype(int), 0, h-1)
        ymax = np.clip(np.ceil(box[:,1].max()).astype(int), 0, h-1)
        mask = np.zeros((ymax-ymin+1, xmax-xmin+1), dtype=np.uint8)
        box[:,0] -= xmin
        box[:,1] -= ymin
        cv2.fillPoly(mask, box.reshape(1,-1,2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]

    def unclip(self, box):
        poly = Polygon(box)
        if poly.area == 0 or poly.length == 0:
            return None
        distance = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        if len(expanded) == 0:
            return None
        return np.array(expanded[0])

    def __call__(self, pred, src_shape):
        if isinstance(pred, np.ndarray) and pred.ndim == 4:
            pred = pred[0,0,:,:]
        segmentation = pred > self.thresh
        src_h, src_w = src_shape
        boxes, scores = self.boxes_from_bitmap(pred, segmentation, src_w, src_h)
        return boxes, scores
# 文字检测
class Det:
    def __init__(
            self,
            image:Image,
            model_path="models/PP-OCRv5_server_det/inference.onnx",
            providers=['DmlExecutionProvider', 'CPUExecutionProvider'],
            debug=False
        ):
        self.image = image
        self.model_path = model_path
        self.providers = providers
        self.debug = debug
    # 文字检测推理前的储存处理
    def resize_norm_img(self,img, max_side_len=960):
        h, w, _ = img.shape
        ratio = 1.0
        if max(h, w) > max_side_len:
            ratio = max_side_len / max(h, w)
        resize_h = int(h*ratio)
        resize_w = int(w*ratio)
        resize_h = max(32, resize_h//32*32)
        resize_w = max(32, resize_w//32*32)
        resized = cv2.resize(img, (resize_w, resize_h)).astype("float32") / 255.0
        mean = np.array([0.485,0.456,0.406], dtype=np.float32)
        std = np.array([0.229,0.224,0.225], dtype=np.float32)
        resized = (resized - mean)/std
        ratio_h = resize_h / h
        ratio_w = resize_w / w
        return resized, (ratio_h, ratio_w)
    def sort_boxes(self,boxes, y_threshold=10):
        """
        按从上到下、从左到右排序
        y_threshold：允许的同一行的 y 差值（像素）
        """
        # 先计算每个 box 的中心点
        boxes = [np.array(b) for b in boxes]
        box_centers = [np.mean(b, axis=0) for b in boxes]

        # 将 box 和中心点绑定
        indexed_boxes = list(zip(boxes, box_centers))

        # 先按 y 排序
        indexed_boxes.sort(key=lambda x: x[1][1])

        # 按行分组
        lines = []
        current_line = [indexed_boxes[0]]
        for i in range(1, len(indexed_boxes)):
            # 如果当前 box 的 y 和上一行差不多 -> 认为在同一行
            if abs(indexed_boxes[i][1][1] - current_line[-1][1][1]) < y_threshold:
                current_line.append(indexed_boxes[i])
            else:
                # 新的一行
                lines.append(current_line)
                current_line = [indexed_boxes[i]]
        if current_line:
            lines.append(current_line)

        # 行内再按 x 坐标排序
        sorted_boxes = []
        for line in lines:
            line.sort(key=lambda x: x[1][0])  # 按中心点 x 排
            sorted_boxes.extend([b for b, _ in line])

        return sorted_boxes

    # 执行预测
    def predict(self):
        contents = []
        # 转为nparray
        img = np.array(self.image) 
        src_h, src_w = img.shape[:2]

        # 缩放图片为960尺寸
        resized_img, (ratio_h, ratio_w) = self.resize_norm_img(img)
        input_tensor = resized_img.transpose(2,0,1)[np.newaxis, :]  # NCHW
        # 执行推理
        session = ort.InferenceSession(self.model_path, providers=self.providers)
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
        pred = outputs[0]  

        # 检测后处理
        # 文字区域分割
        post = DBPostProcess(thresh=0.2, box_thresh=0.2)  # 阈值可调
        boxes, scores = post(pred, (src_h, src_w))
        # boxes = sorted(boxes, key=lambda b: min(p[1] for p in b))
        if len(boxes)>0:
            boxes = self.sort_boxes(boxes, y_threshold=10)
        # 文字区域标记
        # vis = img.copy()
        # for box in boxes:
        #     pts = np.array(box, dtype=np.int32)
        #     cv2.polylines(vis, [pts], True, (0,255,0), 2)

        # 识别每行文字
        for box in boxes:
            pts = np.array(box, dtype=np.float32)
            x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
            crop = img[y:y+h+5, x:x+w]
            if crop.size == 0:
                continue

            rec = Rec(image=crop)
            text = rec.predict()
            contents.append(text)
        return "".join(contents)