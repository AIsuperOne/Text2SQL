import chromadb
import hashlib
from typing import List, Dict, Any, Tuple
from config.settings import CHROMA_DB_PATH

class LocalChromaDB:
    def __init__(self):
        # ChromaDB 1.0+ 版本的初始化方式
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(name="nl2sql")
    
    def _generate_id(self, text):
        """根据文本内容生成唯一ID（哈希）"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def add_embedding(self, text, embedding, metadata=None):
        """添加向量到数据库"""
        doc_id = self._generate_id(text)
        
        # 检查是否已存在
        try:
            existing = self.collection.get(ids=[doc_id])
            if existing and existing['ids']:
                print(f"文档已存在，跳过: {text[:50]}...")
                return
        except:
            pass
        
        # 添加新文档 - ChromaDB 1.0+ 的正确调用方式
        self.collection.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata or {}]
        )
    
    def search(self, embedding, top_k=5):
        """向量搜索（返回文档列表）"""
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k
            )
            
            # 返回文档列表
            if results and results.get("documents") and len(results["documents"]) > 0:
                return results["documents"][0]
            return []
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
    
    def search_with_metadata(self, embedding, top_k=5) -> Tuple[List[str], List[Dict]]:
        """向量搜索（返回文档和元数据）"""
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k
            )
            
            if results and results.get("documents") and len(results["documents"]) > 0:
                docs = results["documents"][0]
                metas = results.get("metadatas", [[]])[0]
                # 确保返回相同长度的列表
                if len(metas) < len(docs):
                    metas.extend([{}] * (len(docs) - len(metas)))
                return docs, metas
            return [], []
        except Exception as e:
            print(f"搜索失败: {e}")
            return [], []

    def has_document(self, text):
        """基于ID查重"""
        doc_id = self._generate_id(text)
        try:
            result = self.collection.get(ids=[doc_id])
            return bool(result and result.get('ids') and len(result['ids']) > 0)
        except Exception as e:
            print(f"检查文档存在性时出错: {e}")
            return False
    
    def get_collection_info(self):
        """获取集合信息（用于调试）"""
        try:
            count = self.collection.count()
            return {"count": count}
        except Exception as e:
            return {"error": str(e)}
    
    def clear_all(self):
        """清空所有数据"""
        try:
            # 方法1：删除并重新创建collection
            self.client.delete_collection("nl2sql")
            self.collection = self.client.create_collection("nl2sql")
            print("ChromaDB已清空")
            return True
        except Exception as e:
            print(f"清空ChromaDB失败: {e}")
            return False
    
    def get_all_documents(self):
        """获取所有文档（用于调试）"""
        try:
            results = self.collection.get()
            return results
        except Exception as e:
            print(f"获取文档失败: {e}")
            return None





