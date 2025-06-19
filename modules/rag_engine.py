from modules.llm_manager import ClaudeLLM
from modules.embedding_manager import QwenEmbedding
from modules.rerank_manager import QwenReranker
from modules.vector_store import LocalChromaDB
from modules.db_connector import MySQLConnector
import pandas as pd
import json

class RAGEngine:
    def __init__(self):
        self.llm = ClaudeLLM()
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
    
    def generate_sql_only(self, question):
        """只生成SQL，不执行"""
        try:
            # 1. 向量化问题
            print(f"[RAG] 向量化问题: {question}")
            q_embed = self.embedder.embed(question)
            print(f"[RAG] 向量维度: {len(q_embed)}")
            
            # 2. 检索相关文档
            print("[RAG] 检索相关文档...")
            docs = self.vector_db.search(q_embed, top_k=10)
            
            # 尝试获取元数据
            try:
                docs_with_meta, metadatas = self.vector_db.search_with_metadata(q_embed, top_k=10)
                use_metadata = True
            except:
                metadatas = [{}] * len(docs)
                use_metadata = False
            
            if not docs:
                print("[RAG] 未找到相关文档，使用默认prompt")
                docs = []
                metadatas = []
            
            # 3. 构建上下文
            context_parts = []
            sql_examples = []
            
            if use_metadata:
                for i, (doc, meta) in enumerate(zip(docs, metadatas)):
                    if meta and meta.get('type') == 'qa' and 'sql' in meta:
                        sql_examples.append(f"问题: {doc}\nSQL: {meta['sql']}")
                    elif meta and meta.get('type') == 'ddl':
                        context_parts.append(f"表结构: {doc}")
                    else:
                        context_parts.append(str(doc))
            else:
                # 如果没有元数据，只使用文档内容
                context_parts = [str(doc) for doc in docs[:5]]
            
            # 4. 构建prompt
            system_prompt = f"""You are a MySQL database SQL expert responsible for converting natural language questions into precise executable SQL queries.

[Database Environment]
- Database: newdbone
- Primary tables: btsbase (alias b) and kpibase (alias k)
- Join condition: b.ID = k.ID

[Core Dimensions]
- Time dimension: k.`开始时间`
- Geographic dimensions: b.`省份`, b.`地市`, b.`区县`, b.`乡镇`, b.`村区`
- Frequency band dimension: b.`frequency_band`

[Context Information]
{chr(10).join(context_parts[:3]) if context_parts else "No specific context available"}

[SQL Examples]
{chr(10).join(sql_examples[:3]) if sql_examples else "No examples available"}

[SQL Generation Rules]
1. Aggregation first: Aggregate raw counters with SUM() before performing divisions or other operations
2. Identifier quoting: All Chinese field names and aliases must be enclosed in backticks
3. GROUP BY clause: Must include all non-aggregated fields from SELECT
4. Numeric precision: All calculated metrics retain 2 decimal places using ROUND(..., 2)
5. Default sorting: By time dimension ascending unless user specifies otherwise
6. Conditional filtering: Build WHERE clause based on user requirements

[Output Requirements]
Return only pure SQL statement text without any explanations, comments, or Markdown formatting marks.

User question: {question}"""
            
            # 5. 生成SQL
            print("[RAG] 生成SQL...")
            sql = self.llm.generate_sql(system_prompt, question)
            print(f"[RAG] 生成的SQL: {sql}")
            
            # 清理SQL（去除可能的markdown标记）
            sql = sql.strip()
            if sql.startswith("```"):
                sql = sql.split("```")[1]
                if sql.startswith("sql"):
                    sql = sql[3:].strip()
            
            return {"sql": sql}
            
        except Exception as e:
            print(f"[RAG] SQL生成错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"sql": "-- SQL生成失败", "error": str(e)}
    
    def ask(self, question):
        """生成SQL并执行"""
        try:
            # 1. 先生成SQL
            result = self.generate_sql_only(question)
            if "error" in result:
                return result
            
            sql = result["sql"]
            
            # 2. 执行SQL
            print("[RAG] 执行SQL...")
            try:
                df = self.db.execute_query(sql)
                return {"sql": sql, "result": df}
            except Exception as e:
                print(f"[RAG] SQL执行失败: {e}")
                return {"sql": sql, "result": pd.DataFrame(), "error": str(e)}
            
        except Exception as e:
            print(f"[RAG] 总体错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "sql": "-- 生成失败", 
                "result": pd.DataFrame(), 
                "error": str(e)
            }



