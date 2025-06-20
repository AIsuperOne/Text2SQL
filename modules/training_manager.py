import yaml
import os
from modules.embedding_manager import QwenEmbedding
from modules.vector_store import LocalChromaDB

class BatchTrainer:
    def __init__(self):
        self.embedder = QwenEmbedding()
        self.vector_db = LocalChromaDB()

    def train_from_metrics_yaml(self, yaml_path):
        """
        从指标定义YAML文件中加载并训练，优化检索精度。
        每个指标会被拆分成多个独立的、专注的文档。
        """
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"警告: 指标定义文件未找到于 {yaml_path}")
            return 0
        except Exception as e:
            print(f"读取或解析YAML文件失败: {e}")
            return 0

        metrics = data.get('metrics', [])
        if not metrics:
            print("YAML文件中没有找到'metrics'列表。")
            return 0
            
        training_count = 0
        for metric in metrics:
            name = metric.get('name', '')
            formula = metric.get('formula', '')
            description = metric.get('description', '')
            unit = metric.get('unit', '')

            if not name or not formula:
                continue

            # 为每个指标创建多个专注的文档，并附带元数据
            # 1. 指标名称本身
            doc1_text = name
            meta1 = {"type": "metric_name", "name": name, "formula": formula}
            if not self.vector_db.has_document(doc1_text):
                embedding1 = self.embedder.embed(doc1_text)
                self.vector_db.add_embedding(doc1_text, embedding1, metadata=meta1)
                training_count += 1

            # 2. 指标名称 + 公式（最重要）
            doc2_text = f"指标 {name} 的计算公式是：{formula}"
            meta2 = {"type": "metric_formula", "name": name, "formula": formula}
            if not self.vector_db.has_document(doc2_text):
                embedding2 = self.embedder.embed(doc2_text)
                self.vector_db.add_embedding(doc2_text, embedding2, metadata=meta2)
                training_count += 1
            
            # 3. 指标名称 + 描述
            doc3_text = f"指标 {name} 的含义是 {description}，单位是 {unit}。"
            meta3 = {"type": "metric_description", "name": name}
            if not self.vector_db.has_document(doc3_text):
                embedding3 = self.embedder.embed(doc3_text)
                self.vector_db.add_embedding(doc3_text, embedding3, metadata=meta3)
                training_count += 1


        print(f"从YAML文件训练了 {training_count} 个新文档。")
        return training_count


    def train_from_ddl(self, ddl_list):
        success_count = 0
        for ddl in ddl_list:
            try:
                embedding = self.embedder.embed(ddl)
                self.vector_db.add_embedding(ddl, embedding, metadata={"type": "ddl"})
                success_count += 1
            except Exception as e:
                print(f"训练DDL失败: {e}")
        return success_count

    def train_from_docs(self, doc_list):
        success_count = 0
        for doc in doc_list:
            try:
                embedding = self.embedder.embed(doc)
                self.vector_db.add_embedding(doc, embedding, metadata={"type": "doc"})
                success_count += 1
            except Exception as e:
                print(f"训练文档失败: {e}")
        return success_count

    def train_from_qa_pairs(self, qa_pairs):
        success_count = 0
        for pair in qa_pairs:
            try:
                q = pair["question"]
                sql = pair["sql"]
                embedding = self.embedder.embed(q)
                self.vector_db.add_embedding(q, embedding, metadata={"type": "qa", "sql": sql})
                success_count += 1
            except Exception as e:
                print(f"训练问答对失败: {e}")
        return success_count

    def train_all(self, ddl_list, doc_list, qa_pairs):
        ddl_count = self.train_from_ddl(ddl_list)
        doc_count = self.train_from_docs(doc_list)
        qa_count = self.train_from_qa_pairs(qa_pairs)
        return {
            "ddl": ddl_count,
            "doc": doc_count,
            "qa": qa_count
        }

    def train_incremental(self, ddl_list, doc_list, qa_pairs):
        counts = {"ddl": 0, "doc": 0, "qa": 0}
        
        for ddl in ddl_list:
            if not self.vector_db.has_document(ddl):
                try:
                    embedding = self.embedder.embed(ddl)
                    self.vector_db.add_embedding(ddl, embedding, metadata={"type": "ddl"})
                    counts["ddl"] += 1
                except Exception as e:
                    print(f"训练DDL失败: {e}")
                    
        for doc in doc_list:
            if not self.vector_db.has_document(doc):
                try:
                    embedding = self.embedder.embed(doc)
                    self.vector_db.add_embedding(doc, embedding, metadata={"type": "doc"})
                    counts["doc"] += 1
                except Exception as e:
                    print(f"训练文档失败: {e}")
                    
        for pair in qa_pairs:
            q = pair["question"]
            sql = pair["sql"]
            if not self.vector_db.has_document(q):
                try:
                    embedding = self.embedder.embed(q)
                    self.vector_db.add_embedding(q, embedding, metadata={"type": "qa", "sql": sql})
                    counts["qa"] += 1
                except Exception as e:
                    print(f"训练问答对失败: {e}")
                    
        return counts



