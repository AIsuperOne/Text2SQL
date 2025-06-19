from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Pie, Scatter, HeatMap, Grid
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
import traceback
import re

class PlotExecutor:
    def __init__(self):
        # pyecharts 全局配置
        self.theme = ThemeType.LIGHT
        
    def execute_plot_code(self, code, df):
        """执行作图代码并返回 pyecharts HTML"""
        
        # 修复常见的 pyecharts 参数错误
        code = self._fix_common_errors(code)
        
        # 验证并修复代码语法
        code = self._fix_syntax_errors(code)
        
        # 准备执行环境
        namespace = {
            'df': df,
            'pd': pd,
            'np': np,
            'opts': opts,
            'Bar': Bar,
            'Line': Line,
            'Pie': Pie,
            'Scatter': Scatter,
            'HeatMap': HeatMap,
            'Grid': Grid,
            'JsCode': JsCode,
            'datetime': datetime
        }
        
        try:
            # 执行代码
            exec(code, namespace)
            
            # 获取生成的图表对象
            if 'chart' in namespace:
                chart = namespace['chart']
                
                # 设置图表宽度为响应式
                chart.width = "100%"
                
                # 渲染为 HTML
                html = chart.render_embed()
                
                return {
                    'success': True,
                    'html': html,
                    'error': None,
                    'fixed_code': code  # 返回修复后的代码
                }
            else:
                return {
                    'success': False,
                    'error': "代码执行完成，但未找到 'chart' 变量。请确保将图表对象赋值给 'chart'。",
                    'html': None
                }
                
        except SyntaxError as e:
            # 尝试自动修复语法错误
            fixed_code = self._auto_fix_syntax(code, str(e))
            if fixed_code != code:
                # 递归调用，尝试执行修复后的代码
                return self.execute_plot_code(fixed_code, df)
            else:
                return {
                    'success': False,
                    'error': f"语法错误: {str(e)}",
                    'traceback': traceback.format_exc(),
                    'html': None
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'html': None
            }
    
    def _fix_common_errors(self, code):
        """修复常见的 pyecharts 代码错误"""
        # 修复 axis_pointer_opts 错误
        code = code.replace('axis_pointer_opts=', 'trigger=')
        
        # 移除 TooltipOpts 中的 axis_pointer_opts
        pattern = r'tooltip_opts=opts\.TooltipOpts\([^)]*axis_pointer_opts=[^,)]*[,)]'
        code = re.sub(pattern, lambda m: m.group(0).replace('axis_pointer_opts=', 'trigger='), code)
        
        # 确保图表有合适的大小设置
        if '.set_global_opts(' in code and 'init_opts=' not in code:
            # 在创建图表时添加初始化选项
            chart_types = ['Bar()', 'Line()', 'Pie()', 'Scatter()', 'HeatMap()']
            for chart_type in chart_types:
                if chart_type in code:
                    code = code.replace(chart_type, f'{chart_type[:-2]}(init_opts=opts.InitOpts(width="100%", height="600px"))')
                    break
        
        return code
    
    def _fix_syntax_errors(self, code):
        """修复常见的语法错误"""
        # 计算括号平衡
        open_parens = code.count('(')
        close_parens = code.count(')')
        
        # 如果括号不平衡，尝试修复
        if open_parens > close_parens:
            # 添加缺失的闭括号
            missing = open_parens - close_parens
            code = code + ')' * missing
        
        # 检查是否有未完成的链式调用
        lines = code.split('\n')
        fixed_lines = []
        in_chain = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # 检查是否是链式调用的开始
            if 'chart = (' in line:
                in_chain = True
            
            # 如果在链式调用中，确保正确的缩进
            if in_chain and stripped.startswith('.'):
                # 确保前面有内容
                if i > 0 and not fixed_lines[-1].strip().endswith(')'):
                    fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
            
            # 检查链式调用是否结束
            if in_chain and ')' in line and not stripped.startswith('.'):
                in_chain = False
        
        return '\n'.join(fixed_lines)
    
    def _auto_fix_syntax(self, code, error_msg):
        """尝试自动修复语法错误"""
        # 如果是括号未闭合错误
        if "'(' was never closed" in error_msg or "')' was never closed" in error_msg:
            # 尝试平衡括号
            lines = code.split('\n')
            
            # 查找可能的问题行
            for i in range(len(lines)):
                line = lines[i]
                if 'chart = (' in line:
                    # 找到链式调用的结束位置
                    j = i + 1
                    while j < len(lines) and (lines[j].strip().startswith('.') or lines[j].strip() == ''):
                        j += 1
                    
                    # 在链式调用结束后添加闭括号
                    if j - 1 < len(lines) and not lines[j-1].strip().endswith(')'):
                        lines[j-1] = lines[j-1].rstrip() + ')'
                    
                    return '\n'.join(lines)
        
        return code
    
    def render_chart(self, result):
        """在 Streamlit 中渲染 pyecharts 图表"""
        if result['success'] and result['html']:
            # 如果有修复后的代码，显示提示
            if 'fixed_code' in result and result.get('fixed_code'):
                st.info("代码已自动修复并成功执行")
            
            # 创建一个响应式的容器
            html_code = f"""
            <div style="width: 100%;overflow-x: auto;">
                {result['html']}
            </div>
            <script>
                // 使图表响应式
                window.addEventListener('resize', function() {{
                    var charts = document.querySelectorAll('div[_echarts_instance_]');
                    charts.forEach(function(chart) {{
                        var echartsInstance = echarts.getInstanceByDom(chart);
                        if (echartsInstance) {{
                            echartsInstance.resize();
                        }}
                    }});
                }});
            </script>
            """
            
            # 使用 streamlit 的 HTML 组件显示图表
            components.html(html_code, height=650, width=1500, scrolling=False)
        else:
            st.error(f"图表渲染失败: {result['error']}")
            if 'traceback' in result:
                with st.expander("查看详细错误"):
                    st.code(result['traceback'])
                    
                # 提供修复建议
                if "'(' was never closed" in result['error']:
                    st.warning("提示：代码中可能存在未闭合的括号。请检查链式调用是否正确结束。")
                    st.code("""
# 正确的链式调用格式：
chart = (
    Bar()
    .add_xaxis(...)
    .add_yaxis(...)
    .set_global_opts(...)
)  # 注意这里的闭括号
""", language='python')

