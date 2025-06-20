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
8. ALWAYS initialize chart with proper size, e.g., init_opts=opts.InitOpts(width="1400px", height="600px")
9. For time series, convert datetime to string: df['时间'].dt.strftime('%Y-%m-%d') if datetime column
10. For many categories (>10), use: axislabel_opts=opts.LabelOpts(rotate=45, interval=0)
11. Round numeric values for display: df['column'].round(2).tolist()
12. If a list of selected_metrics is provided, you MUST ONLY plot these metrics as y-axes.

IMPORTANT:
- The final chart object MUST be assigned to 'chart' variable
- Do NOT use axis_pointer_opts in TooltipOpts
- Return ONLY Python code, no explanations"""

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


