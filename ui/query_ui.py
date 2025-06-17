import streamlit as st
import pandas as pd
import plotly.express as px

def render_query_ui(rag_engine):
    """渲染查询界面"""
    st.header("自然语言查询")
    
    # 查询输入
    question = st.text_input("请输入您的问题：", 
                            placeholder="例如：销售额最高的前10个产品是什么？")
    
    if st.button("查询", type="primary"):
        if question:
            with st.spinner("正在生成SQL并执行查询..."):
                try:
                    result = rag_engine.ask(question)
                    
                    # 显示生成的SQL
                    st.code(result["sql"], language="sql")
                    
                    # 显示查询结果
                    st.subheader("查询结果")
                    st.dataframe(result["result"])
                    
                    # 智能图表生成
                    if st.checkbox("生成可视化图表"):
                        render_visualization(result["result"])
                        
                except Exception as e:
                    st.error(f"查询失败：{str(e)}")
