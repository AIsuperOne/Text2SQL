import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.vector_store import LocalChromaDB
from modules.embedding_manager import QwenEmbedding
import numpy as np

def test_vector_store():
    print("测试向量存储功能...")
    
    # 1. 初始化
    vector_db = LocalChromaDB()
    embedder = QwenEmbedding()
    
    # 2. 测试添加
    test_text = "这是一个测试文档"
    print(f"\n1. 生成测试向量...")
    try:
        test_embedding = embedder.embed(test_text)
        print(f"✅ 向量维度: {len(test_embedding)}")
    except:
        # 如果embedding服务不可用，使用随机向量
        test_embedding = np.random.rand(1024).tolist()
        print(f"⚠️ 使用随机向量测试，维度: {len(test_embedding)}")
    
    print(f"\n2. 添加文档到向量库...")
    vector_db.add_embedding(test_text, test_embedding, {"type": "test"})
    print("✅ 添加成功")
    
    # 3. 测试搜索
    print(f"\n3. 测试搜索功能...")
    results = vector_db.search(test_embedding, top_k=5)
    print(f"✅ 搜索结果数量: {len(results)}")
    if results:
        print(f"   第一个结果: {results[0][:50]}...")
    
    # 4. 测试带元数据的搜索
    print(f"\n4. 测试带元数据的搜索...")
    docs, metas = vector_db.search_with_metadata(test_embedding, top_k=5)
    print(f"✅ 文档数量: {len(docs)}, 元数据数量: {len(metas)}")
    if metas and metas[0]:
        print(f"   第一个元数据: {metas[0]}")
    
    # 5. 测试集合信息
    print(f"\n5. 获取集合信息...")
    info = vector_db.get_collection_info()
    print(f"✅ 集合信息: {info}")
    
    print("\n所有测试完成！")

if __name__ == "__main__":
    test_vector_store()
