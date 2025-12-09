from layout import Layout
from det import Det
from table import TableRec
from vlm_req import VlmReq

class OnnxStructure:
    def __init__(
        self,
        pdf_path="",
        debug=False,
        providers=['DmlExecutionProvider', 'CPUExecutionProvider']
    ):
        self.debug = debug
        self.pdf_path =  pdf_path
        self.providers = providers
    
    # 预测结果
    # page 所在页码
    # content_type: 类型
    # content: 内容
    def predict(self):
        pipelineResult = []
        # 版面检测
        layout = Layout(pdf_path=self.pdf_path,debug=self.debug,providers=self.providers)
        layout_results = layout.predict()
        # 遍历版面检测
        for layout_result in layout_results:
            # 文档标题
            if layout_result["label"] == "doc_title":
                det = Det(image=layout_result["image"],debug=self.debug,providers=self.providers)
                res = det.predict()
                pipelineResult.append({
                    "page": layout_result["page"],
                    "content_type": "文書タイトル",
                    "content": res,
                })
            # 段落标题
            elif layout_result["label"] == "paragraph_title":
                det = Det(image=layout_result["image"],debug=self.debug,providers=self.providers)
                res = det.predict()
                pipelineResult.append({
                    "page": layout_result["page"],
                    "content_type": "セクションタイトル",
                    "content": res
                })
            # 文本
            elif layout_result["label"] == "text":
                det = Det(image=layout_result["image"],debug=self.debug,providers=self.providers)
                res = det.predict()
                pipelineResult.append({
                    "page": layout_result["page"],
                    "content_type": "文章",
                    "content": res
                })
            # 摘要
            elif layout_result["label"] == "abstract":
                det = Det(image=layout_result["image"],debug=self.debug,providers=self.providers)
                res = det.predict()
                pipelineResult.append({
                    "page": layout_result["page"],
                    "content_type": "まとめ",
                    "content": res
                })
            # 目录
            elif layout_result["label"] == "content":
                det = Det(image=layout_result["image"],debug=self.debug,providers=self.providers)
                res = det.predict()
                pipelineResult.append({
                    "page": layout_result["page"],
                    "content_type": "目次",
                    "content": res
                })
            # 图/表标题
            elif layout_result["label"] == "figure_title":
                det = Det(image=layout_result["image"],debug=self.debug,providers=self.providers)
                res = det.predict()
                pipelineResult.append({
                    "page": layout_result["page"],
                    "content_type": "図/表のタイトル",
                    "content": res
                })
            # 参考文献
            elif layout_result["label"] == "reference":
                det = Det(image=layout_result["image"],debug=self.debug,providers=self.providers)
                res = det.predict()
                pipelineResult.append({
                    "page": layout_result["page"],
                    "content_type": "参考文献",
                    "content": res
                })
            # 脚注
            elif layout_result["label"] == "footnote":
                det = Det(image=layout_result["image"],debug=self.debug,providers=self.providers)
                res = det.predict()
                pipelineResult.append({
                    "page": layout_result["page"],
                    "content_type": "脚注",
                    "content": res
                })
            # 页眉
            elif layout_result["label"] == "header":
                det = Det(image=layout_result["image"],debug=self.debug,providers=self.providers)
                res = det.predict()
                pipelineResult.append({
                    "page": layout_result["page"],
                    "content_type": "ヘッダ",
                    "content": res
                })
            # 页脚
            elif layout_result["label"] == "footer":
                det = Det(image=layout_result["image"],debug=self.debug,providers=self.providers)
                res = det.predict()
                pipelineResult.append({
                    "page": layout_result["page"],
                    "content_type": "フッター",
                    "content": res
                })
            # 侧栏文本
            elif layout_result["label"] == "aside_text":
                det = Det(image=layout_result["image"],debug=self.debug,providers=self.providers)
                res = det.predict()
                pipelineResult.append({
                    "page": layout_result["page"],
                    "content_type": "サイドバーテキスト",
                    "content": res
                })
            # 参考文献内容
            elif layout_result["label"] == "aside_text":
                det = Det(image=layout_result["image"],debug=self.debug,providers=self.providers)
                res = det.predict()
                pipelineResult.append({
                    "page": layout_result["page"],
                    "content_type": "サイドバーテキスト",
                    "content": res
                })
            # 表格
            elif layout_result["label"] == "table":
                tableRec= TableRec(image=layout_result["image"],debug=self.debug,providers=self.providers)
                res = tableRec.predict()
                pipelineResult.append({
                    "page": layout_result["page"],
                    "content_type": "文章",
                    "content": res
                })
            # 图片
            elif layout_result["label"] == "image":
                req = VlmReq(image=layout_result["image"])
                res = req.request()
                pipelineResult.append({
                    "page": layout_result["page"],
                    "content_type": "写真",
                    "content": res
                })
            # 印章
            elif layout_result["label"] == "seal":
                req = VlmReq(image=layout_result["image"])
                res = req.request()
                pipelineResult.append({
                    "page": layout_result["page"],
                    "content_type": "シール",
                    "content": res
                })
            # 公式
            elif layout_result["label"] == "formula":
                req = VlmReq(image=layout_result["image"])
                res = req.request()
                pipelineResult.append({
                    "page": layout_result["page"],
                    "content_type": "式",
                    "content": res
                })
            # 图表
            elif layout_result["label"] == "chart":
                req = VlmReq(image=layout_result["image"])
                res = req.request()
                pipelineResult.append({
                    "page": layout_result["page"],
                    "content_type": "チャート",
                    "content": res
                })
            # 算法
        return pipelineResult
