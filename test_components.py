# 测试各个组件是否正常工作
from modules.embedding_manager import QwenEmbedding
from modules.vector_store import LocalChromaDB

# 测试嵌入
embedder = QwenEmbedding()
test_embed = embedder.embed("测试文本")
print(f"嵌入成功，维度: {len(test_embed)}")

# 测试向量库
vector_db = LocalChromaDB()
vector_db.add_embedding("测试文档", test_embed, {"type": "test"})
print("向量添加成功")

# 测试搜索
results = vector_db.search(test_embed, top_k=1)
print(f"搜索结果: {results}")
print(f"结果类型: {type(results)}")
if results:
    print(f"第一个结果类型: {type(results[0])}")
