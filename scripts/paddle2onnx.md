paddlex  --paddle2onnx --paddle_model_dir models/PP-DocLayout_plus-L_infer --onnx_model_dir models/PP-DocLayout_plus-L_infer --opset_version 17
python -m paddle2onnx.optimize --input_model models/PP-DocLayout_plus-L_infer/inference.onnx --output_model models/PP-DocLayout_plus-L_infer/inference_fixed.onnx --input_shape_dict "{'x': [1,3,800,800]}"

paddlex  --paddle2onnx --paddle_model_dir models/PP-LCNet_x1_0_table_cls_infer --onnx_model_dir models/PP-LCNet_x1_0_table_cls_infer --opset_version 17
python -m paddle2onnx.optimize --input_model models/PP-LCNet_x1_0_table_cls_infer/inference.onnx --output_model models/PP-LCNet_x1_0_table_cls_infer/inference_fixed.onnx --input_shape_dict "{'x': [1,3,224,224]}"

paddlex  --paddle2onnx --paddle_model_dir models/PP-OCRv5_server_det_infer --onnx_model_dir models/PP-OCRv5_server_det_infer --opset_version 17
python -m paddle2onnx.optimize --input_model models/PP-OCRv5_server_det_infer/inference.onnx --output_model models/PP-OCRv5_server_det_infer/inference_fixed.onnx --input_shape_dict "{'x': [1,3,736,736]}"
onnxsim models/PP-OCRv5_server_det_infer/inference_fixed.onnx models/PP-OCRv5_server_det_infer/inference_sim.onnx

paddlex  --paddle2onnx --paddle_model_dir models/PP-OCRv5_server_rec_infer --onnx_model_dir models/PP-OCRv5_server_rec_infer --opset_version 17
python -m paddle2onnx.optimize --input_model models/PP-OCRv5_server_rec_infer/inference.onnx --output_model models/PP-OCRv5_server_rec_infer/inference_fixed.onnx --input_shape_dict "{'x': [1,3,48,320]}"

paddlex  --paddle2onnx --paddle_model_dir models/RT-DETR-L_wired_table_cell_det_infer --onnx_model_dir models/RT-DETR-L_wired_table_cell_det_infer --opset_version 17
python -m paddle2onnx.optimize --input_model models//RT-DETR-L_wired_table_cell_det_infer/inference.onnx --output_model models//RT-DETR-L_wired_table_cell_det_infer/inference_fixed.onnx --input_shape_dict "{'x': [1,3,640,640]}"

paddlex  --paddle2onnx --paddle_model_dir models/RT-DETR-L_wireless_table_cell_det_infer --onnx_model_dir models/RT-DETR-L_wireless_table_cell_det_infer --opset_version 17
python -m paddle2onnx.optimize --input_model models/RT-DETR-L_wireless_table_cell_det_infer/inference.onnx --output_model models/RT-DETR-L_wireless_table_cell_det_infer/inference_fixed.onnx --input_shape_dict "{'x': [1,3,640,640]}"

paddlex  --paddle2onnx --paddle_model_dir models/SLANet_infer --onnx_model_dir models/SLANet_infer --opset_version 17
python -m paddle2onnx.optimize --input_model models/SLANet_infer --output_model models/SLANet_infer/inference_fixed.onnx --input_shape_dict "{'x': [1,3,480,480]}"
onnxsim models/SLANet_infer/inference_fixed.onnx models/SLANet_infer/inference_sim.onnx