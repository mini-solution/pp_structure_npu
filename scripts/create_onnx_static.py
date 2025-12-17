import os
import copy
import shutil

# 版面检测
# 复制模型及配置文件
src = "models/PP-DocLayout_plus-L_infer/inference.onnx"
dst_dir = "onnx_static/PP-DocLayout_plus-L_infer"
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(src, dst_dir)
src = "models/PP-DocLayout_plus-L_infer/inference.yml"
dst_dir = "onnx_static/PP-DocLayout_plus-L_infer"
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(src, dst_dir)

# 表格分类
# 复制模型及配置文件
src = "models/PP-LCNet_x1_0_table_cls_infer/inference.onnx"
dst_dir = "onnx_static/PP-LCNet_x1_0_table_cls_infer"
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(src, dst_dir)
src = "models/PP-LCNet_x1_0_table_cls_infer/inference.yml"
dst_dir = "onnx_static/PP-LCNet_x1_0_table_cls_infer"
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(src, dst_dir)

# 文本检测
# 复制模型及配置文件
src = "models/PP-OCRv5_server_det_infer/inference.onnx"
dst_dir = "onnx_static/PP-OCRv5_server_det_infer"
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(src, dst_dir)
src = "models/PP-OCRv5_server_det_infer/inference.yml"
dst_dir = "onnx_static/PP-OCRv5_server_det_infer"
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(src, dst_dir)

# 文本识别
# 复制模型和配置文件
src = "models/PP-OCRv5_server_rec_infer/inference.onnx"
dst_dir = "onnx_static/PP-OCRv5_server_rec_infer"
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(src, dst_dir)
src = "models/PP-OCRv5_server_rec_infer/inference.yml"
dst_dir = "onnx_static/PP-OCRv5_server_rec_infer"
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(src, dst_dir)


# 有线表格识别
# 复制模型及配置文件
src = "models/RT-DETR-L_wired_table_cell_det_infer/inference.onnx"
dst_dir = "onnx_static/RT-DETR-L_wired_table_cell_det_infer"
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(src, dst_dir)
src = "models/RT-DETR-L_wired_table_cell_det_infer/inference.yml"
dst_dir = "onnx_static/RT-DETR-L_wired_table_cell_det_infer"
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(src, dst_dir)

# 无线表格识别
# 复制模型及配置文件
src = "models/RT-DETR-L_wireless_table_cell_det_infer/inference.onnx"
dst_dir = "onnx_static/RT-DETR-L_wireless_table_cell_det_infer"
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(src, dst_dir)
src = "models/RT-DETR-L_wireless_table_cell_det_infer/inference.yml"
dst_dir = "onnx_static/RT-DETR-L_wireless_table_cell_det_infer"
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(src, dst_dir)

# 表头识别
# 复制模型及配置文件
src = "models/SLANet_infer/inference.onnx"
dst_dir = "onnx_static/SLANet_infer"
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(src, dst_dir)
src = "models/SLANet_infer/inference.yml"
dst_dir = "onnx_static/SLANet_infer"
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(src, dst_dir)
