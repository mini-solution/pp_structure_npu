# pp_structure_npu
paddleocr with amd gpu/npu

# 如何安装
#### 创建环境
  ```bash
  conda create -n paddleocr python=3.12
  ```
#### 下载最新release文件包
#### 安装依赖
  ```bash
  pip install -r requirement.txt
  ```
# 如何使用
  #### 参考test_pipeline.py
  ```python
  from pipeline import OnnxStructure
  # 解析pdf
  pipeline = OnnxStructure(pdf_path="gin.pdf")
  # 预测结果，输出为对象
  res = pipeline.predict()

  # 参考 test_pipeline.py,结果输出举例
  # [{
  #    "page": 0,
  #    "content_type": "doc_title",
  #    "content": "GinWebFramework"
  # }]


  # OnnxStructure 主要构造函数说明
  # 参数
  # pdf_path pdf文件
  # vlm_api LMStudio接口，为了调用视觉模型接口，实现图片类摘要获取。默认为空
  # vlm_model LMStudio使用的模型。默认为空
  ```

 #### 结果输出
  ```json
  [{
      "page": 0,
      "content_type": "doc_title",
      "content": "GinWebFramework"
  },
  {
      "page": 0,
      "content_type": "text",
      "content": "Go言語のためのフル機能の最速Webフレームワーク。水晶のようにすっきり。"
  }]
  ```
#### 输出字段说明
   - page 所在页码，从0开始
   - content_type 类型(参考类型枚举)
   - content 内容
  
# 类型枚举
- paragraph_title 段落标题 
- image 图像
- text  文本
- number 页码
- abstract 摘要
- content 目录
- figure_title 图/表标题
- formula 公式
- table 表格
- reference 参考文献
- doc_title 文档标题
- footnote 脚注
- header 页眉
- algorithm 算法
- footer 页脚
- seal 印章
- chart 图表
- formula_number 公式编号
- aside_text 侧栏文本
- reference_content 参考文献内容