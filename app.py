import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_pyecharts
from modules.rag_engine import RAGEngine
from modules.training_manager import BatchTrainer
from utils.plot_executor import PlotExecutor

# ----------------- 新的、更简单的图表构建器 -----------------
def build_simple_chart(df: pd.DataFrame, title: str):
    """
    根据给定的数据框，自动选择合适的图表类型并构建图表。
    """
    from pyecharts import options as opts
    from pyecharts.charts import Bar, Line

    if df.empty:
        return None

    df_copy = df.copy()
    cols = df_copy.columns.tolist()
    
    dim_col, time_col, metric_cols = None, None, []

    # 优先寻找时间列
    for col in cols:
        if '时间' in col or '日期' in col:
            try:
                df_copy[col] = pd.to_datetime(df_copy[col])
                time_col = col
                break
            except (ValueError, TypeError):
                pass

    if time_col:
        dim_col = time_col
        metric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df_copy[c])]
    else:
        for col in cols:
            if not pd.api.types.is_numeric_dtype(df_copy[col]):
                dim_col = col
                break
        if dim_col:
            metric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df_copy[c]) and c != dim_col]
        else:
            dim_col = cols[0]
            metric_cols = cols[1:]

    if not dim_col or not metric_cols:
        return None

    global_opts = {
        "title_opts": opts.TitleOpts(title=title, pos_left="center"),
        "toolbox_opts": opts.ToolboxOpts(is_show=True, feature=opts.ToolBoxFeatureOpts(save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(pixel_ratio=2))),
        "tooltip_opts": opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        "datazoom_opts": [opts.DataZoomOpts(type_="slider"), opts.DataZoomOpts(type_="inside")]
    }

    if time_col:
        chart = Line(init_opts=opts.InitOpts(width="100%", height="500px"))
        df_copy = df_copy.sort_values(by=time_col)
        chart.add_xaxis(df_copy[time_col].dt.strftime('%Y-%m-%d %H:%M').tolist())
        
        yaxis_count = 0
        for y_col in metric_cols:
            if yaxis_count > 0: chart.extend_axis(yaxis=opts.AxisOpts(name=y_col, position="right", offset=yaxis_count * 60))
            chart.add_yaxis(y_col, df_copy[y_col].round(4).tolist(), yaxis_index=yaxis_count)
            yaxis_count += 1
        chart.set_global_opts(**global_opts, legend_opts=opts.LegendOpts(pos_left='center'))
    else:
        chart = Bar(init_opts=opts.InitOpts(width="100%", height="500px"))
        chart.add_xaxis(df_copy[dim_col].tolist())
        for y_col in metric_cols:
            chart.add_yaxis(y_col, df_copy[y_col].round(4).tolist())
        chart.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=30)), **global_opts)

    return chart


# ----------------- 应用主逻辑 -----------------
st.set_page_config(page_title="NL2SQL RAG Demo", layout="wide")

# 初始化
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()
if "trainer" not in st.session_state:
    st.session_state.trainer = BatchTrainer()

rag_engine = st.session_state.rag_engine
trainer = st.session_state.trainer

# 初始化所有查询相关的session state
for key in ['generated_sql', 'current_question', 'refined_question', 'query_result', 'query_error', 'analysis_report']:
    if key not in st.session_state:
        st.session_state[key] = "" if key in ['generated_sql', 'current_question', 'refined_question'] else None


# 创建标签页
tabs = st.tabs(["🧑‍💻 自然语言查询", "⚙️ 批量/增量训练"])

# 1. 查询界面
with tabs[0]:
    st.header("TEXT2SQL")
    
    debug_mode = st.checkbox("显示调试信息", value=False)
    
    # --- 三步式布局 ---
    col1, col2, col3 = st.columns(3)

    # Step 1: 输入问题
    with col1:
        st.markdown("### Step 1: 输入问题")
        question = st.text_area(
            "请输入您的业务问题",
            height=150,
            placeholder="如：查询湖北省的基站数量和小区数量",
            key="question_input"
        )
        
        if st.button("AI识别意图", type="primary", use_container_width=True):
            if question.strip():
                with st.spinner("AI正在识别您的查询意图..."):
                    try:
                        # 准备一个可用指标的简单列表（实际应用中可以从数据库或配置中获取）
                        available_metrics = [
                            "数据业务流量", "无线接通率", "无线掉线率", 
                            "上行数据业务流量", "下行数据业务流量", "系统内切换成功率"
                        ]
                        
                        # 调用LLM进行问题解构
                        refined = rag_engine.llm.refine_question(question.strip(), available_metrics)
                        st.session_state.refined_question = refined
                        st.session_state.current_question = question.strip() # 保存原始问题
                        
                        if debug_mode:
                            with st.expander("查看润色过程", expanded=True):
                                st.write("**原始问题：**", st.session_state.current_question)
                                st.write("**结构化后：**")
                                st.text(refined)

                    except Exception as e:
                        st.error(f"问题识别失败：{str(e)}")
                        # 失败时使用原问题作为意图
                        st.session_state.refined_question = f"**核心意图**: {question.strip()}"
            else:
                st.warning("请先输入问题")

    # Step 2: 确认意图并生成
    with col2:
        st.markdown("### Step 2: 确认意图并生成")
        
        # 使用 st.text_area 来显示和编辑润色后的问题，与Step 1格式对齐
        refined_question_editable = st.text_area(
            "AI识别并结构化的意图（可编辑）",
            value=st.session_state.refined_question,
            height=150,
            key="refined_question_editor",
            # 如果没有润色后的问题，则禁用文本框
            disabled=not st.session_state.refined_question
        )
        
        # “生成SQL”按钮，其可用性取决于文本框中是否有内容
        if st.button("生成SQL", type="primary", use_container_width=True, disabled=not refined_question_editable.strip()):
            with st.spinner("正在生成SQL..."):
                try:
                    # 使用最终编辑后的问题来生成SQL
                    final_question_for_sql = refined_question_editable.strip()
                    # 更新session state中的润色问题，以备后续使用
                    st.session_state.refined_question = final_question_for_sql
                    
                    result = rag_engine.generate_sql_only(final_question_for_sql)
                    st.session_state.generated_sql = result["sql"]
                    
                    # 清理后续状态
                    st.session_state.query_result = None
                    st.session_state.query_error = None
                    st.session_state.analysis_report = None
                    st.rerun()
                except Exception as e:
                    st.error(f"SQL生成失败：{str(e)}")



    # Step 3: 编辑并执行
    with col3:
        st.markdown("### Step 3: 编辑并执行SQL")
        
        edited_sql = st.text_area(
            "AI2SQL编辑器",
            value=st.session_state.generated_sql,
            height=150,
            key="sql_editor",
            disabled=not st.session_state.generated_sql
        )
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("▶️ SQL执行", type="primary", use_container_width=True, disabled=not edited_sql.strip()):
                with st.spinner("正在执行查询..."):
                    try:
                        df = rag_engine.db.execute_query(edited_sql)
                        st.session_state.query_result = df
                        st.session_state.query_error = None
                        st.session_state.analysis_report = None
                    except Exception as e:
                        st.session_state.query_result = None
                        st.session_state.query_error = str(e)
        
        with btn_col2:
            if st.button("💾 保存到训练", type="secondary", use_container_width=True, disabled=not edited_sql.strip()):
                if st.session_state.current_question and edited_sql:
                    # ... 保存逻辑 ...
                    st.success("已保存！")
                else:
                    st.warning("缺少原始问题或SQL")

    # --- 后续显示区域 ---
    if st.session_state.query_error:
        st.error(f"❌ SQL执行错误: {st.session_state.query_error}")
    
    if st.session_state.query_result is not None:
        result_df = st.session_state.query_result
        if not result_df.empty:
            st.divider()
            st.subheader("查询结果")
            st.dataframe(result_df, use_container_width=True)
            
            # AI 智能数据可视化
            st.divider()
            st.subheader("🤖 AI 智能数据可视化")

            if st.button("🔬 生成智能分析报告", type="primary"):
                with st.spinner("AI正在执行深度分析..."):
                    try:
                        # --- 数据预处理和摘要生成 ---
                        dimension_cols = ['开始时间', '省份', '地市', '区县', '乡镇', '村区', 'station_name', 'cell_name', 'frequency_band']
                        existing_dims = [col for col in dimension_cols if col in result_df.columns]
                        metric_cols = [col for col in result_df.select_dtypes(include=np.number).columns if col not in existing_dims]
                        
                        pre_analysis_summary = ""
                        overall_avg_series = None
                        if existing_dims and metric_cols:
                            grouped_analysis_df = result_df.groupby(existing_dims)[metric_cols].mean().round(4)
                            overall_avg_series = result_df[metric_cols].mean().round(4)
                            std_dev_series = result_df[metric_cols].std().round(4)
                            coeff_var_series = (std_dev_series / overall_avg_series).abs().round(4)
                            
                            st.session_state.overall_avg_series = overall_avg_series # 保存全局平均值以供后续使用

                            pre_analysis_summary = f"""
**Pre-computed Analysis Summary:**
1. Grouped Averages:
{grouped_analysis_df.to_string()}
2. Overall Averages (for baseline comparison):
{overall_avg_series.to_string()}
3. Coefficient of Variation (Volatility, for metric selection):
{coeff_var_series.to_string()}
"""
                        
                        categorical_summary = ""
                        for col in result_df.select_dtypes(include=['object', 'category']).columns:
                            unique_values = result_df[col].unique()
                            display_values = list(unique_values[:10]) + ['...'] if len(unique_values) > 10 else list(unique_values)
                            categorical_summary += f"- Column '{col}' contains: {display_values}\n"
                        
                        df_info = f"""
**Original Data Info:**
Shape: {result_df.shape}
Columns and Types: {result_df.dtypes.to_dict()}
Categorical Vocabulary: {categorical_summary if categorical_summary else "None"}
"""
                        
                        # 调用LLM进行文本分析
                        st.session_state.analysis_report = rag_engine.llm.analyze_telecom_data(
                            df_info=df_info,
                            pre_analysis_summary=pre_analysis_summary, # <--- 确保这个参数被传递
                            user_question=st.session_state.current_question,
                            query_result_sample=result_df.head(3).to_string()
                        )

                    except Exception as e:
                        st.error(f"数据分析失败: {str(e)}")
                        if debug_mode: st.exception(e)

            # 显示四维分析报告
            if st.session_state.analysis_report:
                report = st.session_state.analysis_report
                
                dimension_keys = ["overview_analysis", "time_series_analysis", "geo_distribution_analysis", "anomaly_diagnosis"]
                
                for i, key in enumerate(dimension_keys):
                    if report.get(key):
                        st.divider()
                        insight = report[key]
                        st.markdown(f"#### {insight.get('title', key.replace('_', ' ').title())}")
                        if insight.get('explanation'):
                            st.caption(insight['explanation'])
                        
                        # --- START: 新的 "提炼-再查询-可视化" 逻辑 ---
                        chart_data_key = f"chart_data_{i}"
                        
                        if st.button(f"📈 生成可视化图表", key=f"gen_chart_{i}"):
                            with st.spinner(f"正在为【{insight.get('title', '分析')}】提炼数据..."):
                                try:
                                    new_question = insight['explanation']
                                    sql_result = rag_engine.generate_sql_only(new_question)
                                    chart_sql = sql_result.get("sql")
                                    
                                    if debug_mode:
                                        with st.expander("查看图表生成的SQL"):
                                            st.code(chart_sql, language='sql')

                                    chart_df = rag_engine.db.execute_query(chart_sql)
                                    st.session_state[chart_data_key] = chart_df

                                except Exception as e:
                                    st.error(f"图表数据生成失败: {e}")
                                    st.session_state[chart_data_key] = pd.DataFrame() # 出错时置为空DF

                        if chart_data_key in st.session_state:
                            chart_df = st.session_state[chart_data_key]
                            if chart_df is not None and not chart_df.empty:
                                chart_obj = build_simple_chart(chart_df, insight.get('title', ''))
                                if chart_obj:
                                    st_pyecharts(chart_obj, height="500px")
                                else:
                                    st.warning("无法为此数据自动生成图表。")
                            elif chart_df is not None: # 如果是空的DataFrame
                                st.info("为验证此分析点查询到的数据为空。")
                        # --- END: 新逻辑 ---

                st.divider()
                if st.button("🔄 重新分析", type="secondary"):
                    st.session_state.analysis_report = None
                    for i in range(len(dimension_keys)):
                        if f"chart_data_{i}" in st.session_state: del st.session_state[f"chart_data_{i}"]
                    st.rerun()



# 2. 批量/增量训练界面
with tabs[1]:
    st.header("批量/增量训练数据上传")
    
    # 初始化变量
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

    # 文档输入
    st.subheader("1. 导入业务文档")
    doc_input = st.text_area("每行一段业务文档内容", height=150, key="doc_input")
    if doc_input:
        doc_list = [x.strip() for x in doc_input.split('\n') if x.strip()]
        if doc_list:
            st.info(f"已输入 {len(doc_list)} 条文档")

    # 问答对文件上传
    st.subheader("2. 导入SQL问答对")
    
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
    total_items = len(doc_list) + len(qa_pairs)
    if total_items > 0:
        st.info(f"📊 待训练数据统计：文档 {len(doc_list)}条，问答对 {len(qa_pairs)}条")
    
    # 开始训练按钮
    if st.button("🚀 开始训练", type="primary", disabled=(total_items == 0)):
        with st.spinner("训练中，请稍候..."):
            try:
                # 注意：这里传入空的DDL列表
                if incremental:
                    counts = trainer.train_incremental([], doc_list, qa_pairs)
                    st.success(f"✅ 增量训练完成！新增：文档 {counts['doc']}条，问答对 {counts['qa']}条")
                else:
                    counts = trainer.train_all([], doc_list, qa_pairs)
                    st.success(f"✅ 批量训练完成！成功：文档 {counts['doc']}条，问答对 {counts['qa']}条")
                
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


# 侧边栏信息（保持不变）
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











