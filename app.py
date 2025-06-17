import streamlit as st
import pandas as pd
from modules.rag_engine import RAGEngine
from modules.training_manager import BatchTrainer
from utils.visualize import auto_visualize

st.set_page_config(page_title="NL2SQL RAG Demo", layout="wide")

# 初始化
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()
if "trainer" not in st.session_state:
    st.session_state.trainer = BatchTrainer()

rag_engine = st.session_state.rag_engine
trainer = st.session_state.trainer

# 创建标签页
tabs = st.tabs(["🧑‍💻 自然语言查询", "⚙️ 批量/增量训练"])

# 1. 查询界面
with tabs[0]:
    st.header("自然语言转SQL查询")
    
    # 添加调试模式开关
    debug_mode = st.checkbox("显示调试信息", value=False)
    
    # 使用 text_input 而不是 text_area
    question = st.text_input("请输入您的业务问题", placeholder="如：查询销量最高的前10个商品")
    
    if st.button("执行查询", type="primary"):
        if not question.strip():
            st.warning("请输入问题！")
        else:
            with st.spinner("正在推理..."):
                try:
                    # 如果开启调试模式，显示中间步骤
                    if debug_mode:
                        with st.expander("调试信息", expanded=True):
                            st.write("1. 向量化问题...")
                            q_embed = rag_engine.embedder.embed(question)
                            st.write(f"向量维度: {len(q_embed)}")
                            
                            st.write("2. 检索文档...")
                            docs = rag_engine.vector_db.search(q_embed, top_k=5)
                            st.write(f"检索到 {len(docs)} 个文档")
                            for i, doc in enumerate(docs[:3]):
                                st.write(f"文档{i+1}: {doc[:100]}...")
                    
                    result = rag_engine.ask(question)
                    
                    # 显示生成的SQL
                    st.subheader("生成的SQL查询")
                    st.code(result["sql"], language="sql")
                    
                    # 检查是否有错误
                    if "error" in result:
                        st.error(f"执行错误: {result['error']}")
                    
                    # 显示查询结果
                    if result["result"] is not None and not result["result"].empty:
                        st.subheader("查询结果")
                        st.dataframe(result["result"])
                        
                        # 自动智能可视化
                        st.subheader("数据可视化")
                        auto_visualize(result["result"])
                    elif "error" not in result:
                        st.info("查询无结果")
                        
                except Exception as e:
                    st.error(f"系统错误：{str(e)}")
                    if debug_mode:
                        st.exception(e)

# 2. 批量/增量训练界面
with tabs[1]:
    st.header("批量/增量训练数据上传")
    
    # 初始化变量 - 这是关键修改！
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
                            if col.get('COLUMN_COMMENT'):  # 使用 get 方法避免 KeyError
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
                            pass  # 忽略获取样例数据的错误
                    
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
    if ddl_input:  # 只有在有输入时才处理
        ddl_list = [x.strip() for x in ddl_input.split('\n') if x.strip()]
        if ddl_list:
            st.info(f"已输入 {len(ddl_list)} 条DDL语句")

    # 文档输入
    st.subheader("2. 导入业务文档")
    doc_input = st.text_area("每行一段业务文档内容", height=150, key="doc_input")
    if doc_input:  # 只有在有输入时才处理
        doc_list = [x.strip() for x in doc_input.split('\n') if x.strip()]
        if doc_list:
            st.info(f"已输入 {len(doc_list)} 条文档")

    # 问答对文件上传
    st.subheader("3. 导入SQL问答对")
    
    # 提供示例格式
    with st.expander("查看文件格式要求"):
        st.write("CSV或Excel文件需要包含以下两列：")
        example_df = pd.DataFrame({
            "question": ["湖北5G网络的5G基站和5G小区数量", "查询各地市的5G基站数"],
            "sql": [
                "SELECT `省份`,COUNT(DISTINCT station_name) AS `5g基站数`, COUNT(DISTINCT cell_name) AS `5g小区数` FROM btsbase WHERE `省份` = '湖北省' GROUP BY `省份`;",
                "SELECT `地市`, COUNT(DISTINCT station_name) AS `5g基站数` FROM btsbase GROUP BY `地市`;"
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
                qa_pairs = []  # 确保即使出错也有值
            else:
                qa_pairs = df_qa[["question", "sql"]].dropna().to_dict("records")
                st.success(f"✅ 已读取 {len(qa_pairs)} 条问答对")
                
                # 显示前几条数据预览
                if st.checkbox("预览数据"):
                    st.dataframe(df_qa.head())
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}")
            qa_pairs = []  # 确保即使出错也有值

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
    st.write("基于 Qwen + ChromaDB + MySQL")
    
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
            "CREATE TABLE kpibase (ID INT PRIMARY KEY, `开始时间` DATETIME, R1001_012 BIGINT, R1001_001 BIGINT, R1034_012 BIGINT, R1034_001 BIGINT, R1039_002 BIGINT, R1039_001 BIGINT, R2032_012 BIGINT, R2032_001 BIGINT, R1012_001 BIGINT, R1012_002 BIGINT, K1009_001 BIGINT, K1009_002 BIGINT)"
        ]
        test_docs = [
            "btsbase表包含5G基站的基础信息，包括基站名称、小区名称、省份、地市、频段等",
            "kpibase表包含5G网络的KPI指标数据，包括各种性能计数器的值",
            "无线接通率计算公式：100 * (R1001_012/R1001_001) * (R1034_012/R1034_001) * (R1039_002/R1039_001)",
            "5G基站数通过COUNT(DISTINCT station_name)统计，5G小区数通过COUNT(DISTINCT cell_name)统计"
        ]
        test_qa = [
            {
                "question": "湖北省5G 网络价值指标", 
                "sql": "select b.`省份`, k.`开始时间`, b.`frequency_band`, SUM(k.R2032_012) / 1e6 as 下行PDCP层业务流量_GB, SUM(k.R2032_001) / 1e6 as 上行PDCP层业务流量_GB, (SUM(k.R1012_001) + SUM(k.R1012_002)) / 1e6 as 总流量_TB, SUM(k.K1009_001) / 4 as VoNR语音话务量, SUM(k.K1009_002) / 4 as ViNR视频话务量 from btsbase b inner join kpibase k on b.ID = k.ID WHERE b.`省份` = '湖北省' group by b.`省份`, k.`开始时间`, b.`frequency_band` order by k.`开始时间`;"
            },
            {
                "question": "湖北省5G 网络性能指标", 
                "sql": "select b.`省份`, k.`开始时间`, b.`frequency_band`, 100 * (SUM(k.R1001_012) / NULLIF(SUM(k.R1001_001), 0)) * (SUM(k.R1034_012) / NULLIF(SUM(k.R1034_001), 0)) * (SUM(k.R1039_002) / NULLIF(SUM(k.R1039_001), 0)) AS 无线接通率, 100 * (SUM(k.R1004_003) - SUM(k.R1004_004)) / NULLIF(SUM(k.R1004_002) + SUM(k.R1004_007) + SUM(k.R1005_012) + SUM(k.R1006_012), 0) AS 无线掉线率 FROM btsbase b INNER JOIN kpibase k ON b.ID = k.ID WHERE b.`省份` = '湖北省' GROUP BY b.`省份`, k.`开始时间`, b.`frequency_band` order by k.`开始时间`;"
            },
            {
                "question": "统计各地市的5G基站数量",
                "sql": "SELECT `地市`, COUNT(DISTINCT station_name) AS `5g基站数` FROM btsbase GROUP BY `地市`;"
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
            st.code("SELECT b.`省份`, k.`开始时间`, 100 * (SUM(k.R1001_012) / NULLIF(SUM(k.R1001_001), 0)) * (SUM(k.R1034_012) / NULLIF(SUM(k.R1034_001), 0)) * (SUM(k.R1039_002) / NULLIF(SUM(k.R1039_001), 0)) AS 无线接通率 FROM btsbase b INNER JOIN kpibase k ON b.ID = k.ID WHERE b.`省份` = '湖北省' GROUP BY b.`省份`, k.`开始时间` ORDER BY k.`开始时间`", language='sql')
            
            st.write("**Q: 统计各地市的5G基站数量**")
            st.code("SELECT `地市`, COUNT(DISTINCT station_name) AS `5g基站数` FROM btsbase GROUP BY `地市`;", language='sql')
            
            st.write("**Q: 查询5G网络流量指标**")
            st.code("SELECT b.`省份`, SUM(k.R2032_012) / 1e6 as 下行流量_GB, SUM(k.R2032_001) / 1e6 as 上行流量_GB FROM btsbase b INNER JOIN kpibase k ON b.ID = k.ID GROUP BY b.`省份`", language='sql')




