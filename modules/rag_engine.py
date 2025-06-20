import os
import yaml
import pandas as pd
import json
import re
from modules.llm_manager import ClaudeLLM
from modules.embedding_manager import QwenEmbedding
from modules.rerank_manager import QwenReranker
from modules.vector_store import LocalChromaDB
from modules.db_connector import MySQLConnector

class RAGEngine:
    def __init__(self):
        self.llm = ClaudeLLM()
        self.embedder = QwenEmbedding()
        self.reranker = QwenReranker()
        self.vector_db = LocalChromaDB()
        self.db = MySQLConnector()
        
        self.metrics_yaml_path = r"C:\Users\Administrator\PYMo\SuperMO\Text2SQL\all_metrics.yaml"
        self.metrics_definitions = self._load_metrics_definitions(self.metrics_yaml_path)
        self.all_kpi_fields = self._extract_all_kpi_fields()

    def _load_metrics_definitions(self, yaml_path):
        if not os.path.exists(yaml_path):
            print(f"警告: 指标定义文件未找到于 {yaml_path}。")
            return {}
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return {metric['name']: metric for metric in data.get('metrics', [])}
        except Exception as e:
            print(f"加载指标定义文件失败: {e}")
            return {}

    def _extract_all_kpi_fields(self):
        """从所有指标公式中提取K和R开头的字段名"""
        fields = set()
        for metric in self.metrics_definitions.values():
            formula = metric.get('formula', '')
            found = re.findall(r'[KR]\d{4}_\d{3}', formula)
            fields.update(found)
        return list(fields)

    def _find_metrics_in_question(self, question):
        """从问题中查找并返回相关的指标定义"""
        found_metrics = []
        # 使用正则表达式查找所有可能的指标名称
        # 这是一个简化的例子，实际可能需要更复杂的NLP技术
        for name, definition in self.metrics_definitions.items():
            # 简单的关键词匹配
            if name in question:
                found_metrics.append(definition)
        return found_metrics

    def _validate_sql_fields(self, sql):
        """验证SQL中的字段是否在允许的列表中"""
        used_fields = set(re.findall(r'k\.([KR]\d{4}_\d{3})', sql))
        allowed_fields = set(self.all_kpi_fields)
        illegal_fields = used_fields - allowed_fields
        if illegal_fields:
            return f"SQL包含未在指标库中定义的字段: {', '.join(illegal_fields)}"
        return None

    def generate_sql_only(self, question):
        """先润色问题，然后根据问题中的指标动态构建强约束Prompt，最后生成SQL"""
        try:
            print(f"[RAG] LLM结构化问题: {question}")
            structured_question = self.llm.rewrite_question(question)
            print(f"[RAG] 结构化结果: {structured_question}")

            relevant_metrics = self._find_metrics_in_question(structured_question)
            
            metric_formulas_context = []
            if relevant_metrics:
                for metric in relevant_metrics:
                    metric_formulas_context.append(f"- {metric['name']}: {metric['formula']}")
                print(f"[RAG] 找到了相关的官方公式: {metric_formulas_context}")
            else:
                print("[RAG] 未在问题中找到明确的指标，将依赖RAG的泛化能力。")

            q_embed = self.embedder.embed(structured_question)
            docs, metadatas = self.vector_db.search_with_metadata(q_embed, top_k=10)
            
            context_parts = []
            sql_examples = []
            for doc, meta in zip(docs, metadatas):
                if meta and meta.get('type') == 'qa' and 'sql' in meta:
                    sql_examples.append(f"问题: {doc}\nSQL: {meta['sql']}")
                elif meta and meta.get('type') == 'ddl':
                    context_parts.append(f"表结构: {doc}")
                else:
                    context_parts.append(str(doc))

            system_prompt = f"""You are an expert MySQL SQL developer. Your task is to generate precise SQL queries from user questions.

[Database Schema]
- `btsbase` (alias b): Base station info (ID, station_name, 省份, 地市, frequency_band, etc.)
- `kpibase` (alias k): KPI metrics (ID, 开始时间, R* and K* counters)
- Join condition: `b.ID = k.ID`

[Available Metric Formulas]
# THIS IS THE GROUND TRUTH. YOU MUST USE THESE FORMULAS EXACTLY AS PROVIDED.
{chr(10).join(metric_formulas_context) if metric_formulas_context else "No specific formulas found for this query. Rely on your general knowledge and provided context."}

[Context Information]
{chr(10).join(context_parts[:3]) if context_parts else "No specific context available"}

[SQL Examples]
{chr(10).join(sql_examples[:3]) if sql_examples else "No examples available"}

[SQL Generation Rules]
1.  **Strict Formula Adherence**: If a metric is listed in [Available Metric Formulas], you **MUST** use the exact formula provided. Do not invent, simplify, or use any other field names.
2.  **Field Validation**: All `k.` fields in the generated SQL must be from the official KPI counter list. Do not use any `k.` field not on this list.
3.  **Aggregation First**: Always aggregate raw counters with `SUM()` before performing other operations.
4.  **Identifier Quoting**: Enclose all Chinese identifiers in backticks (`` ` ``).
5.  **GROUP BY Clause**: Must include all non-aggregated columns from the `SELECT` list.
6.  **Numeric Precision**: Round all final calculated metrics to 2 decimal places using `ROUND(..., 2)`.
7.  **Unit Conversion**: For traffic, use `/ 1e6` for GB. For rates, use `* 100` for percentage.
8.  **No Frequency Band Grouping**: Do not group by `b.frequency_band` unless explicitly asked.

[Output Requirements]
Return only the raw SQL query text. No explanations or markdown.

# User's structured intent (You must generate SQL based on this):
{structured_question}
"""
            print("[RAG] 生成SQL...")
            sql = self.llm.generate_sql(system_prompt, structured_question)
            print(f"[RAG] 生成的SQL: {sql}")
            
            # SQL后处理和验证
            sql = sql.strip()
            if sql.startswith("```"):
                sql = sql.split("```")[1].strip()
                if sql.startswith("sql"):
                    sql = sql[3:].strip()
            
            validation_error = self._validate_sql_fields(sql)
            if validation_error:
                print(f"[VALIDATION] SQL验证失败: {validation_error}")
                # 可以选择返回错误，或者让LLM重新生成
                return {"sql": sql, "error": validation_error, "structured_question": structured_question}

            return {"sql": sql, "structured_question": structured_question}
        except Exception as e:
            print(f"[RAG] SQL生成错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"sql": "-- SQL生成失败", "error": str(e), "structured_question": ""}

    def ask(self, question):
        """生成SQL并执行"""
        try:
            result = self.generate_sql_only(question)
            if "error" in result and result["error"]:
                return result
            sql = result["sql"]
            print("[RAG] 执行SQL...")
            try:
                df = self.db.execute_query(sql)
                result["result"] = df
                return result
            except Exception as e:
                print(f"[RAG] SQL执行失败: {e}")
                result["result"] = pd.DataFrame()
                result["error"] = str(e)
                return result
        except Exception as e:
            print(f"[RAG] 总体错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "sql": "-- 生成失败",
                "result": pd.DataFrame(),
                "error": str(e)
            }




