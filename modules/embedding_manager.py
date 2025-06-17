from openai import OpenAI
from config.settings import EMBEDDING_CONFIG
import numpy as np

class QwenEmbedding:
    def __init__(self):
        self.client = OpenAI(
            api_key=EMBEDDING_CONFIG['api_key'],
            base_url=EMBEDDING_CONFIG['base_url']
        )
        self.model = EMBEDDING_CONFIG['model']

    def embed(self, text):
        try:
            completion = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=EMBEDDING_CONFIG["dimensions"],
                encoding_format=EMBEDDING_CONFIG["encoding_format"]
            )
            embedding = completion.data[0].embedding
            
            # 确保返回的是列表格式
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            print(f"Embedding生成失败: {e}")
            # 返回一个默认向量（用于测试）
            return [0.0] * EMBEDDING_CONFIG["dimensions"]

