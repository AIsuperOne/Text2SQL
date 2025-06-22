from openai import OpenAI
from config.settings import LLM_CONFIG
import os

class ClaudeLLM:
    def __init__(self):
        # OpenRouter 可能需要额外的配置
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
    
    # ==================== 新增的 rewrite_question 方法 ====================
    def rewrite_question(self, user_question, context_info=None):
        """
        利用LLM对用户自然语言问题进行结构化和逻辑化润色
        返回优化后的结构化业务意图描述
        """
        system_prompt = (
            "You are an expert business analyst. Please rewrite the user's natural language question "
            "into a structured, logically clear, and concise business intent in Chinese, suitable for machine understanding. "
            "Preserve all key business information, avoid redundancy. "
            "If there are filters, aggregation, time/geography/frequency conditions, highlight them explicitly. "
            "Only output the rephrased structured business problem, do not repeat or explain the original."
        )
        if context_info:
            user_prompt = f"Context: {context_info}\nOriginal question: {user_question}"
        else:
            user_prompt = f"Original question: {user_question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        structured = self.chat(messages, temperature=0.2, max_tokens=200)
        return structured.strip()
    # =====================================================================
    
    def generate_plot_code(self, df_info, user_question, query_result_sample, selected_metrics=None):
        """生成 pyecharts 作图代码，并支持筛选指标"""
        system_prompt = """
You are an expert Python data visualization specialist using the Pyecharts library. Your task is to generate a single, complete, and executable Pyecharts code snippet.

[Environment & Constraints]
- The data is in a pandas DataFrame named `df`. Do not load or create it.
- Required objects are pre-imported: `Bar`, `Line`, `Pie`, `opts`, `JsCode`, etc.
- The final chart object MUST be assigned to the variable `chart`.
- Output ONLY the Python code, with no explanations or markdown.

[Chart Generation Rules]
1.  **Chart Type Selection**:
    - For time-series data (e.g., columns named '开始时间', '日期'), use a `Line` chart.
    - For categorical comparisons (e.g., '省份', '地市'), use a `Bar` chart.
    - For proportions, use a `Pie` chart.
    - For comparing multiple metrics, use a multi-line plot or a grouped bar chart.

2.  **Chart Configuration (MUST follow)**:
    - **Initialization**: Always initialize the chart with a responsive width and fixed height: `init_opts=opts.InitOpts(width="100%", height="600px")`.
    - **Smooth Lines**: For all `Line` charts, make the curves smooth by setting `is_smooth=True`.
    - **Toolbox**: Always include a toolbox for user interaction: `toolbox_opts=opts.ToolboxOpts(is_show=True)`.
    - **Data Zoom**: For charts with many data points, add a slider for zooming: `datazoom_opts=[opts.DataZoomOpts(type_="slider")]`.
    - **Labels & Titles**: Use clear, Chinese labels for titles, axes, and legends.
    - **Axis Label Rotation**: If x-axis labels are long or numerous (>10), rotate them: `axislabel_opts=opts.LabelOpts(rotate=45)`.

3.  **Y-Axis Handling (Critical)**:
    - **Smart Range**: Set the y-axis `min_` and `max_` to make trends visible. The data plot should occupy roughly 3/4 of the chart's height. For example, if data ranges from 99.5 to 99.8, a good range is `min_=99.0, max_=100`.
    - **Dual Y-Axis**: If plotting two metrics with vastly different scales (e.g., a rate near 100% and another near 0%), you **MUST** use a second Y-axis to prevent overlap. Follow this pattern:
      ```python
      # Dual Y-Axis Example
      line = Line(init_opts=...).add_xaxis(...)
      line.add_yaxis("Metric A (Left)", ...)
      bar = Bar().add_xaxis(...) # or another Line()
      bar.add_yaxis("Metric B (Right)", ...)
      line.overlap(bar)
      line.extend_axis(yaxis=opts.AxisOpts(name="Metric B Name", position="right", ...))
      chart = line
      ```

4.  **Metric Selection**:
    - If a list of `selected_metrics` is provided in the user prompt, you **MUST** plot only those metrics as y-axes.
"""


        user_prompt = f"""Create a pyecharts visualization for this data:

User Question: {user_question}

Dataframe Info:
{df_info}

Sample Data:
{query_result_sample}
"""
        # 如果用户选择了指标，将其加入Prompt
        if selected_metrics:
            user_prompt += f"\nUser has selected the following metrics to plot: {', '.join(selected_metrics)}. You MUST ONLY plot these metrics as y-axes."

        user_prompt += "\nGenerate pyecharts code that creates an appropriate visualization with responsive width.\nRemember to assign the chart to 'chart' variable."

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


