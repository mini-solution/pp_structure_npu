import json
from pipeline import OnnxStructure

# json格式化
def format_json(data, indent=4, ensure_ascii=False):
    return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)

# 解析pdf
pipeline = OnnxStructure(pdf_path="gin.pdf")
res = pipeline.predict()
# 格式化结果
json_res = format_json(res)
# 打印结果
print(json_res)
