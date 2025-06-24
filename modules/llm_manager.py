from openai import OpenAI
from config.settings import LLM_CONFIG
import os

class ClaudeLLM:
    def __init__(self):
        self.client = OpenAI(
            api_key=LLM_CONFIG['api_key'],
            base_url=LLM_CONFIG['base_url'],
            default_headers={
                "HTTP-Referer": "https://openrouter.ai/anthropic/claude-opus-4/api",
                "X-Title": "NL2SQL RAG Demo",
            }
        )
        self.model = LLM_CONFIG['model']

    def chat(self, messages, **kwargs):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"LLM调用失败: {e}")
            raise e

    def generate_sql(self, system_prompt, user_question):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ]
        return self.chat(messages, temperature=0.1, max_tokens=2000)
    
    def refine_question(self, question, available_metrics=None):
        """
        使用LLM解构并优化用户问题，识别维度和指标，并生成结构化的、对人和机器都友好的意图描述。
        """
        # --- 整合点 1: 构建强大且清晰的指标上下文 ---
        # (来自版本2的优点，并进行了优化)
        metrics_context = "No specific metrics list provided for mapping."
        if available_metrics:
            metrics_context = f"""
**Available Key Metrics:**
A list of key metrics available in the database is provided below. Your primary task is to accurately map the user's general terms (like "性能", "流量") to one or more of these specific metric names.
- {', '.join(available_metrics)}

**Mapping Guidelines:**
- "性能" or "质量" -> Map to relevant metrics like `无线接通率`, `无线掉线率`, `系统内切换成功率`.
- "流量" -> Map to `数据业务流量`, `上行数据业务流量`, `下行数据业务流量`.
- **Crucially, DO NOT invent or list any metric that is not in the provided list.**
"""

        # --- 整合点 2: 构建融合了两个版本优点的 System Prompt ---
        system_prompt = f"""You are an expert AI assistant that translates natural language questions about telecom networks into structured, optimized analytical queries. Your goal is to deconstruct the user's question and then rephrase it into a clear, structured summary that is easy for both the user to confirm and for a downstream Text-to-SQL model to understand.

**1. Database Context & Available Dimensions:**
   - **Time Dimension**: `开始时间`
   - **Geographical Dimensions**: `省份`, `地市`, `区县`, `乡镇`, `村区`, `station_name`, `cell_name`
   - **Frequency Band Dimension**: `frequency_band`
   - **Term Clarification**: "基站" means `station_name`, "小区" means `cell_name`.

**2. Available Metrics Context:**
{metrics_context}

**3. Your Task (Two Steps in One):**
   **Step A (Deconstruct):** Mentally break down the user's question. Identify the core intent, the dimensions involved, and the specific metrics requested (mapping them from the list).
   **Step B (Reconstruct & Optimize):** Based on your deconstruction, generate a structured summary in **Chinese**. This summary must be clear, specific, and actionable.

**4. Output Format (Strict Markdown):**
You must follow this markdown format precisely.
**核心意图**: [A clear, one-sentence summary of the user's goal, optimized for clarity. E.g., "对比湖北省内十堰市和咸宁市在指定时间段内的关键网络性能指标。"]
**分析维度**: [List the specific dimensions identified from the user's question, e.g., `省份`, `地市`, `开始时间`]
**关键指标**: [List the specific metrics you mapped from the available list, e.g., `无线掉线率`, `数据业务流量`]

**5. Examples:**
- **User Question**: "查询湖北省十堰市和咸宁市的网络性能"
- **Your Output**:
**核心意图**: 对比湖北省十堰市和咸宁市的网络性能表现。
**分析维度**: `省份`, `地市`
**关键指标**: `无线接通率`, `无线掉线率`, `数据业务流量`

- **User Question**: "上周5G流量统计"
- **Your Output**:
**核心意图**: 统计过去一周(从XX日到XX日)的5G数据业务总量及上下行分布。
**分析维度**: `开始时间`
**关键指标**: `数据业务流量`, `上行数据业务流量`, `下行数据业务流量`

**REMEMBER: Your output is a bridge. It must be human-readable for confirmation and machine-readable (structured) for the next step. Stick to the format.**"""

        # --- 整合点 3: 统一的 messages 和 chat 调用 ---
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        
        # 使用较低的 temperature 保证输出的稳定性和格式遵循度
        return self.chat(messages, temperature=0.1, max_tokens=600)

    
    def analyze_telecom_data(self, df_info, pre_analysis_summary, user_question, query_result_sample):
        """
        根据高度智能化的四维分析框架，对通信数据进行结构化分析并生成可视化图表定义。
        """
        system_prompt = """You are a world-class Chinese telecom network data analyst. Your task is to generate a highly insightful and structured analysis report based on the provided pre-computed summary.

**CRITICAL DIRECTIVE: Your entire analysis MUST be derived from the `Pre-computed Analysis Summary`. All charts MUST be beautiful, readable, and directly support your findings.**

**Analysis Framework & Charting Rules:**

**1. 网络概览分析 (Overview Analysis):**
   - **Text**: Summarize the key findings from the pre-computed summary.
   - **Chart Definition**:
     - **Identify the highest-level geographical dimension** available (e.g., `地市` if present, otherwise `省份`).
     - Create a **Bar chart** showing the total count or sum of a key metric (e.g., number of stations, total traffic) for each category in that dimension. This is the **汇总趋势图**.

**2. 时序趋势动态分析 (Time-Series Trend Analysis):**
   - **Condition**: Must have a time column.
   - **Text**: Analyze the trends over time.
   - **Chart Definition (Smart Selection)**:
     - **Calculate the Coefficient of Variation (Std Dev / Mean)** for each metric in the pre-computed summary.
     - **If more than 3 metrics exist, select the TOP 2 metrics with the highest Coefficient of Variation** (most volatile/dynamic).
     - Create a **multi-series Line chart** for these selected high-volatility metrics. This focuses the user's attention on the most important trends.

**3. 地理空间分布分析 (Geospatial Distribution Analysis):**
   - **Condition**: Must have a location column.
   - **Text**: Identify the **single most valuable insight** from the geographical data (e.g., which city has the worst performance, which has the best).
   - **Chart Definition**:
     - **Identify the metric that best supports your most valuable insight.**
     - Create a clean **Bar chart** for **only that single metric**, sorted from best to worst (or vice versa), to clearly illustrate your point.

**4. 异常指标识别与诊断 (Anomaly Detection & Diagnosis):**
   - **Text**: Diagnose the most significant anomaly by comparing a specific group's metric to the overall average. Quantify the deviation (e.g., "XX市的掉线率 (0.5%) 是整体平均水平 (0.1%) 的5倍").
   - **Chart Definition**:
     - Create a **Bar chart** comparing the anomalous group's value against the `Overall Average` for that specific metric. This provides a direct visual comparison.

**Output Format (Strict JSON, all text values in Chinese):**
```json
{
  "overview_analysis": { "title": "网络概览分析", "explanation": "...", "chart_definition": { ... } },
  "time_series_analysis": { "title": "...", "explanation": "...", "chart_definition": { ... } } or null,
  "geo_distribution_analysis": { "title": "...", "explanation": "...", "chart_definition": { ... } } or null,
  "anomaly_diagnosis": { "title": "...", "explanation": "...", "chart_definition": { ... } } or null
}
Final Check: Ensure every chart is simple, beautiful, and tells a clear story that matches your text analysis."""
        user_prompt = f"""Analyze the following data strictly based on the pre-computed summary. Respond in Chinese.
User's Original Question: {user_question}

Pre-computed Analysis Summary (Your ONLY Source of Truth):
{pre_analysis_summary}

Original Data Info (For chart definition column names):
{df_info}

Generate the structured analysis report in the specified JSON format."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.chat(messages, temperature=0.3, max_tokens=8000)

        try:
            import json
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            response = response.replace('\\\n', '\n').replace('\\\r\n', '\r\n')
            return json.loads(response)
        except Exception as e:
            print(f"JSON解析或验证失败: {e}")
            return {"overview_analysis": {"explanation": f"AI分析结果解析失败，请重试。错误: {e}\n原始返回: {response}"}}
        


    def generate_plot_code(self, df_info, user_question, query_result_sample):
        """生成 pyecharts 作图代码"""
        system_prompt = """You are a Python data visualization expert specializing in pyecharts. Generate pyecharts code for data visualization.

    Rules:
    1. The dataframe is available as 'df'. DO NOT load or create df.
    2. Import statements are NOT needed. Available objects: Bar, Line, Pie, Scatter, HeatMap, Grid, opts, JsCode
    3. Create appropriate chart based on data:
    - Time series (开始时间/日期): Line chart
    - Categories (省份/地市): Bar chart  
    - Percentages/rates: Bar or Line with formatter
    - Proportions: Pie chart
    - Multiple series: Multi-line or grouped bar
    4. MUST assign the final chart object to variable 'chart'
    5. Use Chinese labels and titles
    6. NEVER use 'width'/'height' parameter in opts.LabelOpts, opts.AxisOpts, or opts.TitleOpts.
    7. Only set width/height in chart's init_opts, e.g. Bar(init_opts=opts.InitOpts(width="100%", height="600px"))
    8. ALWAYS initialize chart with proper size:
    ```python
    # Example with responsive width
    chart = (
        Bar(init_opts=opts.InitOpts(width="100%", height="600px"))
        .add_xaxis(df['category'].tolist())
        .add_yaxis("series_name", df['value'].tolist())
        .set_global_opts(
            title_opts=opts.TitleOpts(title="图表标题"),
            xaxis_opts=opts.AxisOpts(name="X轴名称", axislabel_opts=opts.LabelOpts(rotate=45)),
            yaxis_opts=opts.AxisOpts(name="Y轴名称"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            datazoom_opts=[opts.DataZoomOpts(type_="slider")],
            legend_opts=opts.LegendOpts(pos_left="center", pos_top="top")
        )
    )
For time series, convert datetime to string: df['时间'].dt.strftime('%Y-%m-%d') if datetime column
For many categories (>10), use: axislabel_opts=opts.LabelOpts(rotate=45, interval=0)
Round numeric values: df['column'].round(2).tolist()
Use toolbox for additional features: toolbox_opts=opts.ToolboxOpts()
IMPORTANT:

ALWAYS use init_opts=opts.InitOpts(width="100%", height="600px") when creating chart

The final chart object MUST be assigned to 'chart' variable

Do NOT use axis_pointer_opts in TooltipOpts

Return ONLY Python code, no explanations"""

        user_prompt = f"""Create a pyecharts visualization for this data:

User Question: {user_question}

Dataframe Info:
{df_info}

Sample Data:
{query_result_sample}

Generate pyecharts code that creates an appropriate visualization with responsive width.
Remember to assign the chart to 'chart' variable."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        code = self.chat(messages, temperature=0.3, max_tokens=1500)

        # 清理代码
        code = code.strip()
        if code.startswith("```"):
            lines = code.split('\n')
            start_idx = 0
            end_idx = len(lines)
            
            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    if start_idx == 0:
                        start_idx = i + 1
                    else:
                        end_idx = i
                        break
            
            code = '\n'.join(lines[start_idx:end_idx])

        # 额外的代码清理
        code = code.replace('axis_pointer_opts=', 'trigger=')

        return code




