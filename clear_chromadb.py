import shutil
import os
from config.settings import CHROMA_DB_PATH

def clear_chromadb():
    """清空ChromaDB数据库"""
    if os.path.exists(CHROMA_DB_PATH):
        try:
            shutil.rmtree(CHROMA_DB_PATH)
            print(f"✅ 已成功删除ChromaDB目录: {CHROMA_DB_PATH}")
        except Exception as e:
            print(f"❌ 删除失败: {e}")
    else:
        print(f"ℹ️ ChromaDB目录不存在: {CHROMA_DB_PATH}")

if __name__ == "__main__":
    response = input("确定要清空ChromaDB吗？(yes/no): ")
    if response.lower() == 'yes':
        clear_chromadb()
    else:
        print("已取消操作")
