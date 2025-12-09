import cv2
import onnx
from pipeline import OnnxStructure
from onnx import shape_inference

pipeline = OnnxStructure(pdf_path="gin.pdf")
res = pipeline.predict()
print(res)
# model = onnx.load("models/PP-OCRv5_server_rec/inference.onnx")
# model = shape_inference.infer_shapes(model)
# onnx.save(model, "model_shape.onnx")
# for node in model.graph.node:
#     if node.op_type == "MaxPool":
#         for attr in node.attribute:
#             print(attr)