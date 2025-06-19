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
    7. Only set width/height in chart's init_opts, e.g. Bar(init_opts=opts.InitOpts(width="1400px", height="600px"))
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

