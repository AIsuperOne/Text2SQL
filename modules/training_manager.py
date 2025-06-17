from modules.embedding_manager import QwenEmbedding
from modules.vector_store import LocalChromaDB

class BatchTrainer:
    def __init__(self):
        self.embedder = QwenEmbedding()
        self.vector_db = LocalChromaDB()

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
        
        # DDL增量训练
        for ddl in ddl_list:
            if not self.vector_db.has_document(ddl):
                try:
                    embedding = self.embedder.embed(ddl)
                    self.vector_db.add_embedding(ddl, embedding, metadata={"type": "ddl"})
                    counts["ddl"] += 1
                except Exception as e:
                    print(f"训练DDL失败: {e}")
                    
        # 文档增量训练
        for doc in doc_list:
            if not self.vector_db.has_document(doc):
                try:
                    embedding = self.embedder.embed(doc)
                    self.vector_db.add_embedding(doc, embedding, metadata={"type": "doc"})
                    counts["doc"] += 1
                except Exception as e:
                    print(f"训练文档失败: {e}")
                    
        # 问答对增量训练
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


