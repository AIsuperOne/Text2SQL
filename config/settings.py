import os

LLM_CONFIG = {
    "api_key": os.getenv("DASHSCOPE_API_KEY"),
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "qwen-turbo-latest"
}

EMBEDDING_CONFIG = {
    "api_key": os.getenv("DASHSCOPE_API_KEY"),
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "text-embedding-v4",
    "dimensions": 1024,
    "encoding_format": "float"
}

RERANK_CONFIG = {
    "model": "gte-rerank-v2"
}

MYSQL_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "root1234",
    "database": "newdbone"
}

CHROMA_DB_PATH = "./chroma_db"

