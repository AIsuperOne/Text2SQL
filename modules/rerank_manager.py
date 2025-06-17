import dashscope
from http import HTTPStatus
from config.settings import RERANK_CONFIG
from typing import List, Dict, Any

class QwenReranker:
    def rerank(self, query: str, documents: List[str], top_n: int = 10) -> List[Dict[str, Any]]:
        """
        对文档进行重排序
        返回格式统一为: [{"document": "文本内容", "score": 0.95}, ...]
        """
        try:
            # 确保documents是字符串列表
            doc_list = []
            for doc in documents:
                if isinstance(doc, str):
                    doc_list.append(doc)
                elif isinstance(doc, dict):
                    # 如果是字典，尝试提取文本内容
                    doc_text = doc.get('document', '') or doc.get('text', '') or str(doc)
                    doc_list.append(doc_text)
                else:
                    doc_list.append(str(doc))
            
            # 调用rerank API
            resp = dashscope.TextReRank.call(
                model=RERANK_CONFIG["model"],
                query=query,
                documents=doc_list,
                top_n=min(top_n, len(doc_list)),  # 确保top_n不超过文档数
                return_documents=True
            )
            
            if resp.status_code == HTTPStatus.OK:
                results = resp.output.get("results", [])
                # 标准化返回格式
                formatted_results = []
                for item in results:
                    formatted_results.append({
                        "document": item.get("document", ""),
                        "score": item.get("relevance_score", 0)
                    })
                return formatted_results
            else:
                print(f"Rerank失败: {resp.status_code}, {resp.message}")
                # 返回原始文档作为fallback
                return [{"document": doc, "score": 1.0} for doc in doc_list[:top_n]]
                
        except Exception as e:
            print(f"Rerank异常: {str(e)}")
            # 异常时返回原始文档
            return [{"document": str(doc), "score": 1.0} for doc in documents[:top_n]]
