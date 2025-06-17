import streamlit as st
import plotly.express as px

def auto_visualize(df):
    if df.empty:
        st.info("无数据可展示")
        return
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    category_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(numeric_cols) >= 1 and len(category_cols) >= 1:
        x_col = category_cols[0]
        y_col = numeric_cols[0]
        st.write(f"自动推荐：横轴【{x_col}】，纵轴【{y_col}】")
        fig = px.bar(df, x=x_col, y=y_col)
        st.plotly_chart(fig, use_container_width=True)
    elif len(numeric_cols) >= 2:
        x_col, y_col = numeric_cols[:2]
        st.write(f"自动推荐：散点图（{x_col} vs {y_col}）")
        fig = px.scatter(df, x=x_col, y=y_col)
        st.plotly_chart(fig, use_container_width=True)
    elif len(numeric_cols) == 1:
        st.write(f"自动推荐：直方图（{numeric_cols[0]}）")
        fig = px.histogram(df, x=numeric_cols[0])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("未找到适合可视化的字段。")
