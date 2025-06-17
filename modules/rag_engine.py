from modules.llm_manager import QwenLLM
from modules.embedding_manager import QwenEmbedding
from modules.rerank_manager import QwenReranker
from modules.vector_store import LocalChromaDB
from modules.db_connector import MySQLConnector
import pandas as pd
import json

class RAGEngine:
    def __init__(self):
        self.llm = QwenLLM()
        self.embedder = QwenEmbedding()
        self.reranker = QwenReranker()
        self.vector_db = LocalChromaDB()
        self.db = MySQLConnector()
    
    def _safe_str(self, obj):
        """安全地将任何对象转换为字符串"""
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, dict):
            # 如果是字典，尝试提取有意义的文本
            if 'document' in obj:
                return str(obj['document'])
            elif 'text' in obj:
                return str(obj['text'])
            else:
                # 如果没有特定字段，转换整个字典
                return json.dumps(obj, ensure_ascii=False)
        else:
            return str(obj)
    
    def ask(self, question):
        try:
            # 1. 向量化问题
            print(f"[RAG] 向量化问题: {question}")
            q_embed = self.embedder.embed(question)
            print(f"[RAG] 向量维度: {len(q_embed)}")
            
            # 2. 检索相关文档
            print("[RAG] 检索相关文档...")
            docs = self.vector_db.search(q_embed, top_k=10)
            
            if not docs:
                print("[RAG] 未找到相关文档，使用默认prompt")
                system_prompt = "你是一个SQL专家。请根据用户问题生成合适的SQL查询。"
                sql = self.llm.generate_sql(system_prompt, question)
                df = self.db.execute_query(sql)
                return {"sql": sql, "result": df}
            
            print(f"[RAG] 检索到 {len(docs)} 个文档")
            
            # 3. 将所有文档转换为字符串
            doc_strings = []
            for i, doc in enumerate(docs):
                doc_str = self._safe_str(doc)
                doc_strings.append(doc_str)
                print(f"[RAG] 文档{i+1}类型: {type(doc)}, 转换后长度: {len(doc_str)}")
            
            # 4. Rerank - 使用try-except保护
            context_docs = []
            try:
                print("[RAG] 开始重排序...")
                reranked = self.reranker.rerank(question, doc_strings, top_n=3)
                print(f"[RAG] Rerank返回类型: {type(reranked)}")
                
                # 处理rerank结果
                for item in reranked:
                    doc_text = self._safe_str(item)
                    if doc_text:
                        context_docs.append(doc_text)
                
                print(f"[RAG] 重排序后文档数: {len(context_docs)}")
                
            except Exception as e:
                print(f"[RAG] Rerank失败: {e}, 使用原始文档前3个")
                context_docs = doc_strings[:3]
            
            # 5. 确保有上下文文档
            if not context_docs:
                print("[RAG] 无有效上下文，使用原始文档")
                context_docs = doc_strings[:3]
            
            # 6. 构建Prompt - 使用编号列表格式
            context_parts = []
            for i, doc in enumerate(context_docs):
                # 再次确保是字符串
                doc_str = self._safe_str(doc)
                # 截断过长的文档
                if len(doc_str) > 500:
                    doc_str = doc_str[:500] + "..."
                context_parts.append(f"{i+1}. {doc_str}")
            
            # 使用安全的join操作
            try:
                context = "\n".join(context_parts)
            except Exception as e:
                print(f"[RAG] Join失败: {e}")
                # 如果join失败，逐个处理
                context = ""
                for part in context_parts:
                    context += str(part) + "\n"
            
            system_prompt = f"""你是一个SQL专家。请根据以下信息生成SQL查询：

相关信息：
{context}

用户问题：{question}

请生成对应的SQL查询语句。只返回SQL语句，不要有其他解释。"""
            
            # 7. 生成SQL
            print("[RAG] 生成SQL...")
            sql = self.llm.generate_sql(system_prompt, question)
            print(f"[RAG] 生成的SQL: {sql}")
            
            # 8. 执行SQL
            print("[RAG] 执行SQL...")
            try:
                df = self.db.execute_query(sql)
                return {"sql": sql, "result": df}
            except Exception as e:
                print(f"[RAG] SQL执行失败: {e}")
                # 返回空结果而不是抛出异常
                return {"sql": sql, "result": pd.DataFrame(), "error": str(e)}
            
        except Exception as e:
            print(f"[RAG] 总体错误: {str(e)}")
            import traceback
            traceback.print_exc()
            # 返回错误信息而不是抛出异常
            return {
                "sql": "-- 生成失败", 
                "result": pd.DataFrame(), 
                "error": str(e)
            }


