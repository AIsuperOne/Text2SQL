import streamlit as st
import pandas as pd
from modules.rag_engine import RAGEngine
from modules.training_manager import BatchTrainer
from utils.plot_executor import PlotExecutor
import base64
import os

st.set_page_config(page_title="NL2SQL RAG Demo", layout="wide")

# 初始化
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()
if "trainer" not in st.session_state:
    st.session_state.trainer = BatchTrainer()

rag_engine = st.session_state.rag_engine
trainer = st.session_state.trainer

# 初始化查询相关的session state
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'structured_question' not in st.session_state:
    st.session_state.structured_question = ""
if 'generated_sql' not in st.session_state:
    st.session_state.generated_sql = ""
if 'query_result' not in st.session_state:
    st.session_state.query_result = None
if 'query_error' not in st.session_state:
    st.session_state.query_error = None
if 'plot_code' not in st.session_state:
    st.session_state.plot_code = ""
if 'plot_result' not in st.session_state:
    st.session_state.plot_result = None

# 创建标签页
tabs = st.tabs(["🧑‍💻 自然语言查询", "⚙️ 批量/增量训练"])

# 1. 查询界面
with tabs[0]:
    st.header("自然语言转SQL查询")
    
    debug_mode = st.checkbox("显示调试信息", value=False)
    
    # ==================== 1. 问题输入与润色 ====================
    st.subheader("Step 1: 输入您的问题")
    
    question = st.text_input(
        "请输入您的业务问题", 
        placeholder="例如：湖北省十堰市最后一天的数据业务流量大于5，无线掉线率指标从大到小排序TOP10",
        key="user_question_input"
    )
    
    if st.button("🔍 润色问题", type="primary", disabled=not question.strip()):
        with st.spinner("AI正在理解并润色您的问题..."):
            st.session_state.current_question = question
            result = rag_engine.rewrite_question_only(question)
            
            if result.get('error'):
                st.error(f"问题润色失败: {result['error']}")
                st.session_state.structured_question = question # 失败则使用原问题
            else:
                st.session_state.structured_question = result['structured_question']
            
            # 清空后续步骤的状态
            st.session_state.generated_sql = ""
            st.session_state.query_result = None
            st.session_state.query_error = None
            st.session_state.plot_code = ""
            st.session_state.plot_result = None
    
    # ==================== 2. 润色后问题确认与SQL生成 ====================
    if st.session_state.structured_question:
        st.divider()
        st.subheader("Step 2: 确认意图并生成SQL")
        
        edited_structured_question = st.text_area(
            "AI理解并润色后的问题（可修改）：",
            value=st.session_state.structured_question,
            height=100,
            key="structured_question_editor"
        )
        
        if st.button("生成SQL", disabled=not edited_structured_question.strip()):
            with st.spinner("正在生成SQL..."):
                try:
                    result = rag_engine.generate_sql_only(edited_structured_question)
                    st.session_state.generated_sql = result["sql"]
                    st.session_state.structured_question = edited_structured_question
                    
                except Exception as e:
                    st.error(f"SQL生成失败：{str(e)}")
                    if debug_mode:
                        st.exception(e)

    # ==================== 3. SQL编辑与执行 ====================
    if st.session_state.generated_sql:
        st.divider()
        st.subheader("Step 3: 编辑并执行SQL")
        
        st.info(f"原始问题：{st.session_state.current_question}")
        st.success(f"润色后问题：{st.session_state.structured_question}")
        
        edited_sql = st.text_area(
            "生成的SQL（可编辑）：",
            value=st.session_state.generated_sql,
            height=200,
            key="sql_editor"
        )
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
        
        with col1:
            execute_button = st.button("▶️ 执行SQL", type="secondary")
        with col2:
            save_to_training = st.button("💾 保存到训练", type="secondary")
        with col3:
            clear_button = st.button("🗑️ 清空", type="secondary")
        
        if execute_button:
            with st.spinner("正在执行查询..."):
                try:
                    df = rag_engine.db.execute_query(edited_sql)
                    st.session_state.query_result = df
                    st.session_state.query_error = None
                except Exception as e:
                    st.session_state.query_result = None
                    st.session_state.query_error = str(e)
        
        if save_to_training:
            if st.session_state.structured_question and edited_sql:
                try:
                    qa_pair = {
                        "question": st.session_state.structured_question,
                        "sql": edited_sql
                    }
                    count = trainer.train_from_qa_pairs([qa_pair])
                    if count > 0:
                        st.success(f"✅ 已将问答对保存到训练数据！")
                        with st.expander("查看保存的训练数据"):
                            st.write(f"**问题：** {qa_pair['question']}")
                            st.code(qa_pair['sql'], language='sql')
                    else:
                        st.warning("该问答对可能已存在于训练数据中")
                except Exception as e:
                    st.error(f"保存失败：{str(e)}")
            else:
                st.warning("请确保有问题和SQL语句")
        
        if clear_button:
            st.session_state.generated_sql = ""
            st.session_state.current_question = ""
            st.session_state.structured_question = ""
            st.session_state.query_result = None
            st.session_state.query_error = None
            st.session_state.plot_code = ""
            st.session_state.plot_result = None
            st.rerun()
        
        if st.session_state.query_error:
            st.error(f"❌ SQL执行错误: {st.session_state.query_error}")
        
        if st.session_state.query_result is not None:
            result_df = st.session_state.query_result
            if not result_df.empty:
                st.divider()
                st.subheader("查询结果")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("返回行数", len(result_df))
                with col2:
                    st.metric("列数", len(result_df.columns))
                with col3:
                    csv = result_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 下载CSV",
                        data=csv,
                        file_name=f"query_result_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                st.dataframe(result_df, use_container_width=True)
                
                st.divider()
                st.subheader("🤖 AI 智能数据可视化")
                
                numeric_cols = [col for col in result_df.columns if pd.api.types.is_numeric_dtype(result_df[col])]
                default_metrics = numeric_cols[:2] if len(numeric_cols) > 1 else numeric_cols
                selected_metrics = st.multiselect(
                    "请选择要展示的指标（可多选）",
                    options=numeric_cols,
                    default=default_metrics
                )
                
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    if st.button("🎨 生成图表", type="primary", disabled=not selected_metrics):
                        with st.spinner("AI正在分析数据并生成图表代码..."):
                            try:
                                df_info = f"""
Shape: {result_df.shape}
Columns: {list(result_df.columns)}
Data types:
{result_df.dtypes.to_string()}
"""
                                sample_data = result_df.head(5).to_string()
                                plot_code = rag_engine.llm.generate_plot_code(
                                    df_info,
                                    st.session_state.structured_question,
                                    sample_data,
                                    selected_metrics=selected_metrics
                                )
                                
                                st.session_state.plot_code = plot_code
                                st.session_state.plot_result = None
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"生成图表代码失败: {str(e)}")
                
                if st.session_state.plot_code:
                    st.write("**生成的 Pyecharts 代码：**")
                    
                    edited_code = st.text_area(
                        "可以编辑代码后运行：",
                        value=st.session_state.plot_code,
                        height=300,
                        key="plot_code_editor"
                    )
                    
                    if st.button("▶️ 运行代码", type="secondary"):
                        with st.spinner("正在生成图表..."):
                            executor = PlotExecutor()
                            result = executor.execute_plot_code(edited_code, result_df)
                            st.session_state.plot_result = result
                            
                            if result['success']:
                                st.success("✅ 图表生成成功！")
                                executor.render_chart(result)
                            else:
                                st.error(f"❌ 图表生成失败: {result['error']}")
                                if 'traceback' in result:
                                    with st.expander("查看详细错误"):
                                        st.code(result['traceback'])
                
                # Pyecharts 示例
                with st.expander("📚 Pyecharts 代码示例"):
                    st.markdown("""
                    **柱状图示例：**
                    ```python
                    chart = (
                        Bar()
                        .add_xaxis(df['地市'].tolist())
                        .add_yaxis("基站数量", df['基站数量'].tolist())
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="各地市基站数量"),
                            xaxis_opts=opts.AxisOpts(name="地市", axislabel_opts=opts.LabelOpts(rotate=45)),
                            yaxis_opts=opts.AxisOpts(name="基站数量"),
                            datazoom_opts=opts.DataZoomOpts(type_="slider")
                        )
                        .set_series_opts(
                            label_opts=opts.LabelOpts(is_show=True, position="top")
                        )
                    )
                    ```
                    
                    **折线图示例：**
                    ```python
                    # 处理时间格式
                    df['时间'] = pd.to_datetime(df['开始时间'])
                    df = df.sort_values('时间')
                    
                    chart = (
                        Line()
                        .add_xaxis(df['时间'].dt.strftime('%Y-%m-%d').tolist())
                        .add_yaxis(
                            "无线接通率", 
                            df['无线接通率'].round(2).tolist(),
                            markpoint_opts=opts.MarkPointOpts(
                                data=[opts.MarkPointItem(type_="max"), opts.MarkPointItem(type_="min")]
                            )
                        )
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="无线接通率趋势"),
                            xaxis_opts=opts.AxisOpts(name="日期", axislabel_opts=opts.LabelOpts(rotate=45)),
                            yaxis_opts=opts.AxisOpts(name="接通率(%)", min_=95),
                            tooltip_opts=opts.TooltipOpts(trigger="axis"),
                            datazoom_opts=[opts.DataZoomOpts(type_="slider", range_start=0, range_end=100)],
                            is_smooth=True,  # 设置为平滑曲线
                            toolbox_opts=opts.ToolboxOpts(is_show=True)  # 工具箱
                        )
                    )
                    ```
                    
                    **饼图示例：**
                    ```python
                    chart = (
                        Pie()
                        .add(
                            "流量占比",
                            [list(z) for z in zip(df['省份'].tolist(), df['数据业务流量'].tolist())],
                            radius=["40%", "75%"]
                        )
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="各省份流量占比"),
                            legend_opts=opts.LegendOpts(orient="vertical", pos_left="left")
                        )
                        .set_series_opts(
                            label_opts=opts.LabelOpts(formatter="{b}: {c} GB ({d}%)")
                        )
                    )
                    ```
                                
                    **双Y轴图表示例 (重要):**
                    ```python
                    # 准备X轴数据
                    x_data = df['开始时间'].dt.strftime('%Y-%m-%d').tolist()

                    # 主Y轴（接通率，范围98-100）
                    line_chart = (
                        Line(init_opts=opts.InitOpts(width="1400px", height="600px"))
                        .add_xaxis(x_data)
                        .add_yaxis(
                            "无线接通率",
                            df['无线接通率'].round(2).tolist(),
                            is_smooth=True,
                            label_opts=opts.LabelOpts(is_show=False),
                        )
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="网络性能指标趋势 (双Y轴)"),
                            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                            toolbox_opts=opts.ToolboxOpts(is_show=True),
                            legend_opts=opts.LegendOpts(pos_left='center'),
                            yaxis_opts=opts.AxisOpts(
                                name="接通率 (%)",
                                min_=98, # 智能设置Y轴范围
                                max_=100,
                                axislabel_opts=opts.LabelOpts(formatter="{value} %"),
                            ),
                        )
                        .extend_axis( # 扩展第二个Y轴
                            yaxis=opts.AxisOpts(
                                name="掉线率 (%)",
                                position="right",
                                min_=0,
                                max_=1, # 智能设置Y轴范围
                                axislabel_opts=opts.LabelOpts(formatter="{value} %"),
                            )
                        )
                    )

                    # 次Y轴（掉线率，范围0-1）
                    bar_chart = (
                        Line()
                        .add_xaxis(x_data)
                        .add_yaxis(
                            "无线掉线率",
                            df['无线掉线率'].round(2).tolist(),
                            yaxis_index=1, # 指定使用第二个Y轴
                            is_smooth=True,
                            label_opts=opts.LabelOpts(is_show=False),
                        )
                    )

                    # 将两个图表重叠
                    line_chart.overlap(bar_chart)
                    chart = line_chart
                    ```
                                                    
                    **多系列折线图：**
                    ```python
                    chart = (
                        Line()
                        .add_xaxis(df['时间'].dt.strftime('%Y-%m-%d').tolist())
                        .add_yaxis("无线接通率", df['无线接通率'].round(2).tolist())
                        .add_yaxis("无线掉线率", df['无线掉线率'].round(2).tolist(), yaxis_index=1)
                        .extend_axis(
                            yaxis=opts.AxisOpts(
                                name="掉线率(%)",
                                position="right"
                            )
                        )
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="数据业务网络性能指标趋势"),
                            tooltip_opts=opts.TooltipOpts(trigger="axis"),
                            datazoom_opts=[opts.DataZoomOpts()],
                            is_smooth=True,  # 设置为平滑曲线
                            toolbox_opts=opts.ToolboxOpts(is_show=True)  # 工具箱
                        )
                    )
                    ```
                    """)
            else:
                st.info("查询返回空结果")
    
    # SQL模板示例
    with st.expander("💡 SQL查询示例"):
        st.markdown("""
        **常用查询模板：**
        
        1. **基站统计**
        ```sql
        SELECT b.`省份`, COUNT(DISTINCT b.station_name) AS `基站数量` 
        FROM btsbase b 
        GROUP BY b.`省份`
        ORDER BY `基站数量` DESC
        ```
        
        2. **性能指标查询**
        ```sql
            SELECT b.`省份`, k.`开始时间`,
                round(100 *  (SUM(k.R1001_012) / NULLIF(SUM(k.R1001_001), 0)) * (SUM(k.R1034_012) / NULLIF(SUM(k.R1034_001), 0)) * (SUM(k.R1039_002) / NULLIF(SUM(k.R1039_001), 0)) ,2)  AS `无线接通率`
            FROM btsbase b 
            INNER JOIN kpibase k ON b.ID = k.ID 
            WHERE b.`省份` = '湖北省'
            GROUP BY b.`省份`, k.`开始时间`
            ORDER BY k.`开始时间`
        ```
        
        3. **流量统计**
        ```sql
        SELECT b.`地市`, 
            ROUND(SUM(k.R1012_001 + k.R1012_002) / 1024 / 1024, 2) AS `数据业务流量`
        FROM btsbase b 
        INNER JOIN kpibase k ON b.ID = k.ID
        GROUP BY b.`地市`
        ORDER BY `数据业务流量` DESC
        ```
        """)

# 2. 批量/增量训练界面
with tabs[1]:
    st.header("批量/增量训练数据上传")
    
    # 初始化变量
    ddl_list = []
    doc_list = []
    qa_pairs = []
    
    # 新增：自动导入数据库结构
    st.subheader("0. 自动导入数据库结构")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("从已连接的MySQL数据库自动导入表结构信息")
    with col2:
        if st.button("🔄 导入数据库结构", type="secondary"):
            with st.spinner("正在读取数据库结构..."):
                try:
                    # 获取数据库结构
                    schema_info = rag_engine.db.get_schema_info()
                    
                    # 准备训练数据
                    auto_ddl_list = []
                    auto_doc_list = []
                    
                    for table_info in schema_info:
                        # 添加DDL
                        auto_ddl_list.append(table_info['ddl'])
                        
                        # 添加表说明文档
                        table_doc = f"表名: {table_info['table_name']}"
                        if table_info['table_comment']:
                            table_doc += f", 说明: {table_info['table_comment']}"
                        
                        # 添加列说明
                        col_docs = []
                        for col in table_info['columns']:
                            if col.get('COLUMN_COMMENT'):
                                col_docs.append(f"{col['COLUMN_NAME']}: {col['COLUMN_COMMENT']}")
                        
                        if col_docs:
                            table_doc += ", 字段说明: " + "; ".join(col_docs)
                        
                        auto_doc_list.append(table_doc)
                        
                        # 获取样例数据
                        try:
                            sample_df = rag_engine.db.get_sample_data(table_info['table_name'], limit=3)
                            if not sample_df.empty:
                                sample_doc = f"表{table_info['table_name']}的样例数据: {sample_df.to_string()}"
                                auto_doc_list.append(sample_doc)
                        except:
                            pass
                    
                    # 执行训练
                    counts = trainer.train_incremental(auto_ddl_list, auto_doc_list, [])
                    
                    st.success(f"✅ 成功导入 {len(schema_info)} 个表的结构信息！")
                    
                    # 显示导入的表
                    with st.expander("查看导入的表"):
                        for table_info in schema_info:
                            st.write(f"- {table_info['table_name']}: {table_info.get('table_comment', '')}")
                    
                except Exception as e:
                    st.error(f"导入失败: {str(e)}")
    
    st.divider()

    # DDL输入
    st.subheader("1. 导入 DDL 语句")
    ddl_input = st.text_area("每行一个DDL建表语句", height=150, key="ddl_input")
    if ddl_input:
        ddl_list = [x.strip() for x in ddl_input.split('\n') if x.strip()]
        if ddl_list:
            st.info(f"已输入 {len(ddl_list)} 条DDL语句")

    # 文档输入
    st.subheader("2. 导入业务文档")
    doc_input = st.text_area("每行一段业务文档内容", height=150, key="doc_input")
    if doc_input:
        doc_list = [x.strip() for x in doc_input.split('\n') if x.strip()]
        if doc_list:
            st.info(f"已输入 {len(doc_list)} 条文档")

    # 问答对文件上传
    st.subheader("3. 导入SQL问答对")
    
    # 提供示例格式
    with st.expander("查看文件格式要求"):
        st.write("CSV或Excel文件需要包含以下两列：")
        example_df = pd.DataFrame({
            "question": ["湖北省网络的基站数量和小区数量", "查询各地市的基站数量"],
            "sql": [
                "SELECT `省份`,COUNT(DISTINCT station_name) AS `基站数量`, COUNT(DISTINCT cell_name) AS `小区数量` FROM btsbase WHERE `省份` = '湖北省' GROUP BY `省份`;",
                "SELECT `地市`, COUNT(DISTINCT station_name) AS `基站数量` FROM btsbase GROUP BY `地市`;"
            ]
        })
        st.dataframe(example_df)
    
    qa_file = st.file_uploader("上传问答对文件（csv/xlsx）", type=["csv", "xlsx"])
    
    if qa_file:
        try:
            if qa_file.name.endswith(".csv"):
                df_qa = pd.read_csv(qa_file)
            else:
                df_qa = pd.read_excel(qa_file)
                
            if "question" not in df_qa.columns or "sql" not in df_qa.columns:
                st.error("❌ 文件必须包含 'question' 和 'sql' 两列")
                qa_pairs = []
            else:
                qa_pairs = df_qa[["question", "sql"]].dropna().to_dict("records")
                st.success(f"✅ 已读取 {len(qa_pairs)} 条问答对")
                
                # 显示前几条数据预览
                if st.checkbox("预览数据"):
                    st.dataframe(df_qa.head())
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}")
            qa_pairs = []

    # 训练选项
    st.divider()
    incremental = st.checkbox("仅增量训练（跳过已入库内容）", value=True)
    
    # 显示待训练数据统计
    total_items = len(ddl_list) + len(doc_list) + len(qa_pairs)
    if total_items > 0:
        st.info(f"📊 待训练数据统计：DDL {len(ddl_list)}条，文档 {len(doc_list)}条，问答对 {len(qa_pairs)}条")
    
    # 开始训练按钮
    if st.button("🚀 开始训练", type="primary", disabled=(total_items == 0)):
        with st.spinner("训练中，请稍候..."):
            try:
                if incremental:
                    counts = trainer.train_incremental(ddl_list, doc_list, qa_pairs)
                    st.success(f"✅ 增量训练完成！新增：DDL {counts['ddl']}条，文档 {counts['doc']}条，问答对 {counts['qa']}条")
                else:
                    counts = trainer.train_all(ddl_list, doc_list, qa_pairs)
                    st.success(f"✅ 批量训练完成！成功：DDL {counts['ddl']}条，文档 {counts['doc']}条，问答对 {counts['qa']}条")
                
                # 显示向量库状态
                try:
                    db_info = trainer.vector_db.get_collection_info()
                    st.info(f"📚 向量库当前文档总数: {db_info.get('count', 'N/A')}")
                except:
                    pass
                
            except Exception as e:
                st.error(f"❌ 训练失败：{str(e)}")
                st.exception(e)
    
    # 清空按钮
    if total_items > 0:
        if st.button("🗑️ 清空输入"):
            st.rerun()

# 侧边栏信息
with st.sidebar:
    st.title("系统信息")
    st.write("NL2SQL RAG Demo")
    st.write("基于 Claude + ChromaDB + MySQL")
    
    # 显示系统状态
    st.divider()
    st.subheader("系统状态")
    try:
        db_info = trainer.vector_db.get_collection_info()
        st.metric("向量库文档数", db_info.get('count', 0))
    except:
        st.metric("向量库文档数", "未连接")
    
    # 添加测试数据按钮
    st.divider()
    if st.button("添加测试数据"):
        test_ddl = [
            "CREATE TABLE btsbase (ID INT PRIMARY KEY, station_name VARCHAR(100), cell_name VARCHAR(100), `省份` VARCHAR(50), `地市` VARCHAR(50), frequency_band VARCHAR(20))",
            "CREATE TABLE kpibase (ID INT PRIMARY KEY, `开始时间` DATETIME, R1001_012 BIGINT, R1001_001 BIGINT, R1034_012 BIGINT, R1034_001 BIGINT, R1039_002 BIGINT, R1039_001 BIGINT, R2032_012 BIGINT, R2032_001 BIGINT, R1012_001 BIGINT, R1012_002 BIGINT, K1009_001 BIGINT, K1009_002 BIGINT, R1004_002 BIGINT, R1004_003 BIGINT, R1004_004 BIGINT, R1004_007 BIGINT, R1005_012 BIGINT, R1006_012 BIGINT)"
        ]
        test_docs = [
            "btsbase表包含5G基站的基础信息，包括基站名称、小区名称、省份、地市、频段等",
            "kpibase表包含5G网络的KPI指标数据，包括各种性能计数器的值",
            "无线接通率计算公式：100 * (R1001_012/R1001_001) * (R1034_012/R1034_001) * (R1039_002/R1039_001)",
            "无线掉线率计算公式：100 * (R1004_003 - R1004_004) / (R1004_002 + R1004_007 + R1005_012 + R1006_012)",
            "基站数量通过COUNT(DISTINCT station_name)统计，小区数量通过COUNT(DISTINCT cell_name)统计"
        ]
        test_qa = [
            {
                "question": "查询湖北省的基站数量",
                "sql": "SELECT '湖北省' AS `省份`, COUNT(DISTINCT station_name) AS `基站数量` FROM btsbase WHERE `省份` = '湖北省'"
            },
            {
                "question": "统计各地市的基站数量",
                "sql": "SELECT `地市`, COUNT(DISTINCT station_name) AS `基站数量` FROM btsbase GROUP BY `地市` ORDER BY `基站数量` DESC"
            },
            {
                "question": "查询湖北省的数据业务的网络性能指标",
                "sql": "select b.`省份`, k.`开始时间`, round(100 * (SUM(k.R1001_012) / nullif(SUM(k.R1001_001), 0)) * (SUM(k.R1034_012) / nullif(SUM(k.R1034_001), 0)) * (SUM(k.R1039_002) / nullif(SUM(k.R1039_001), 0)), 2) as 无线接通率, round(100 * (SUM(k.R1004_003) - SUM(k.R1004_004)) / nullif(SUM(k.R1004_002) + SUM(k.R1004_007) + SUM(k.R1005_012) + SUM(k.R1006_012), 0), 2) as 无线掉线率, round(100 * SUM(k.R2007_002 + k.R2007_004 + k.R2006_004 + k.R2006_008 + k.R2005_004 + k.R2005_008) / nullif(SUM(k.R2007_001 + k.R2007_003 + k.R2006_001 + k.R2006_005 + k.R2005_001 + k.R2005_005), 0), 2) as 系统内切换成功率, round(100 * SUM(k.R2075_001 + k.R2040_014) / nullif(SUM(k.R2034_033), 0), 2) as EPSFallbackVoLTE回落成功率 from btsbase b inner join kpibase k on b.ID = k.ID group by b.`省份`, k.`开始时间` order by k.`开始时间`;"
            },
            {
                "question": "查询湖北省数据业务流量指标",
                "sql": "select b.`省份`, k.`开始时间`, round((SUM(k.R1012_001) + SUM(k.R1012_002)),2) / 1e6 AS 数据业务流量, round(SUM(k.R2032_012) / 1e6,2) AS 下行数据业务流量, round(SUM(k.R2032_001) / 1e6,2) AS 上行数据业务流量 FROM btsbase b INNER JOIN kpibase k ON b.ID = k.ID GROUP BY b.`省份`, k.`开始时间` order by k.`开始时间`;"
            },
            {
                "question": "查询湖北省的VONR的网络性能指标",
                "sql": "select b.`省份`, k.`开始时间`, round(100 * (SUM(k.R1034_013) / NULLIF(SUM(k.R1034_002), 0)) * SUM(k.R1001_018 + k.R1001_015) / NULLIF(SUM(k.R1001_007 + k.R1001_004), 0), 2) AS VoNR无线接通率, round(100 * SUM(k.R2035_003 - k.R2035_013) / NULLIF(SUM(k.R2035_003 + k.R2035_026), 0), 2) AS VoNR语音掉线率, round(100 * SUM(k.R2005_063 + k.R2005_067 + k.R2006_071 + k.R2006_075 + k.R2007_036 + k.R2007_040) / NULLIF(SUM(k.R2005_060 + k.R2005_064 + k.R2006_068 + k.R2006_072 + k.R2007_033 + k.R2007_037), 0), 2) AS VoNR系统内切换成功率 FROM btsbase b INNER JOIN kpibase k ON b.ID = k.ID GROUP BY b.`省份`, k.`开始时间` order by k.`开始时间`;"
            }
        ]
        
        with st.spinner("添加测试数据..."):
            try:
                counts = trainer.train_all(test_ddl, test_docs, test_qa)
                st.success("测试数据添加完成！")
            except Exception as e:
                st.error(f"添加测试数据失败: {str(e)}")
    
    # 添加通信行业常见查询
    if st.button("查看通信行业示例"):
        with st.expander("示例查询", expanded=True):
            st.write("**Q: 查询湖北省的无线接通率趋势**")
            st.code("""SELECT b.`省份`, k.`开始时间`, 
    ROUND(100 * (SUM(k.R1001_012) / NULLIF(SUM(k.R1001_001), 0)) * 
    (SUM(k.R1034_012) / NULLIF(SUM(k.R1034_001), 0)) * 
    (SUM(k.R1039_002) / NULLIF(SUM(k.R1039_001), 0)), 2) AS `无线接通率`
FROM btsbase b 
INNER JOIN kpibase k ON b.ID = k.ID 
WHERE b.`省份` = '湖北省' 
GROUP BY b.`省份`, k.`开始时间` 
ORDER BY k.`开始时间`""", language='sql')
            
            st.write("**Q: 统计各地市的基站数量**")
            st.code("SELECT `地市`, COUNT(DISTINCT station_name) AS `基站数量` FROM btsbase GROUP BY `地市` ORDER BY `基站数量` DESC", language='sql')
            
            st.write("**Q: 查询数据业务流量指标**")
            st.code("""SELECT b.`省份`, 
    ROUND(SUM(k.R2032_012) / 1e6, 2) as `数据业务下行流量`, 
    ROUND(SUM(k.R2032_001) / 1e6, 2) as `数据业务上行流量`,
    ROUND((SUM(k.R1012_001) + SUM(k.R1012_002)) / 1024 / 1024, 2) as `数据业务流量`
FROM btsbase b 
INNER JOIN kpibase k ON b.ID = k.ID 
GROUP BY b.`省份`
ORDER BY `数据业务流量` DESC""", language='sql')
    
    # 清空向量库功能
    st.divider()
    st.subheader("数据库管理")
    if st.button("🗑️ 清空向量库", type="secondary"):
        confirm = st.checkbox("确认清空所有训练数据", key="confirm_clear")
        if confirm:
            try:
                # 清空向量数据库
                trainer.vector_db.clear_all()
                st.success("向量库已清空！")
                st.rerun()
            except Exception as e:
                st.error(f"清空失败: {str(e)}")
        else:
            st.warning("请勾选确认框以清空向量库")







