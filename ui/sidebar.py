import streamlit as st
from config.settings import AppConfig
from modules.llm_manager import LLMManager
from modules.vector_store import VectorStoreManager
from modules.db_connector import DatabaseManager
from modules.rag_engine import RAGEngine
from ui.sidebar import render_sidebar
from ui.training_ui import render_training_ui
from ui.query_ui import render_query_ui

def main():
    st.set_page_config(page_title="NL2SQL Assistant", layout="wide")
    
    # 初始化session state
    if "config" not in st.session_state:
        st.session_state.config = AppConfig()
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None
        
    # 渲染侧边栏配置
    render_sidebar()
    
    # 主界面标签页
    tab1, tab2, tab3 = st.tabs(["🔧 配置", "📚 训练", "💬 查询"])
    
    with tab1:
        st.header("系统配置")
        # 配置界面实现
        
    with tab2:
        if st.session_state.rag_engine:
            render_training_ui(st.session_state.rag_engine)
        else:
            st.warning("请先完成系统配置")
            
    with tab3:
        if st.session_state.rag_engine:
            render_query_ui(st.session_state.rag_engine)
        else:
            st.warning("请先完成系统配置和训练")

if __name__ == "__main__":
    main()
