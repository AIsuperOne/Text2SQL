class PlotTemplates:
    """常用作图模板"""
    
    @staticmethod
    def time_series_template():
        return """
# 时间序列图
plt.figure(figsize=(12, 6))

# 确保时间列是datetime类型
if '时间' in df.columns:
    df['时间'] = pd.to_datetime(df['时间'])
    x_col = '时间'
elif '开始时间' in df.columns:
    df['开始时间'] = pd.to_datetime(df['开始时间'])
    x_col = '开始时间'
else:
    x_col = df.columns[0]

# 找到数值列
numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) > 0:
    y_col = numeric_cols[0]
    
    # 绘制线图
    plt.plot(df[x_col], df[y_col], marker='o', linewidth=2, markersize=6)
    
    # 设置标题和标签
    plt.title(f'{y_col}趋势图', fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    
    # 旋转x轴标签
    plt.xticks(rotation=45)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 添加平均线
    mean_val = df[y_col].mean()
    plt.axhline(y=mean_val, color='r', linestyle='--', alpha=0.7, label=f'平均值: {mean_val:.2f}')
    plt.legend()

plt.tight_layout()
"""
    
    @staticmethod
    def bar_chart_template():
        return """
# 柱状图
plt.figure(figsize=(10, 6))

# 找到类别列和数值列
cat_cols = df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(include=['number']).columns

if len(cat_cols) > 0 and len(num_cols) > 0:
    x_col = cat_cols[0]
    y_col = num_cols[0]
    
    # 如果数据太多，只显示前20个
    plot_df = df.nlargest(20, y_col) if len(df) > 20 else df
    
    # 创建柱状图
    bars = plt.bar(range(len(plot_df)), plot_df[y_col])
    
    # 设置颜色渐变
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # 设置x轴标签
    plt.xticks(range(len(plot_df)), plot_df[x_col], rotation=45, ha='right')
    
    # 添加数值标签
    for i, v in enumerate(plot_df[y_col]):
        plt.text(i, v + v*0.01, f'{v:.1f}', ha='center', va='bottom')
    
    # 设置标题和标签
    plt.title(f'{x_col}的{y_col}分布', fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)

plt.tight_layout()
"""
    
    @staticmethod
    def pie_chart_template():
        return """
# 饼图
plt.figure(figsize=(10, 8))

# 找到类别列和数值列
cat_cols = df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(include=['number']).columns

if len(cat_cols) > 0 and len(num_cols) > 0:
    label_col = cat_cols[0]
    value_col = num_cols[0]
    
    # 如果类别太多，只显示前10个
    plot_df = df.nlargest(10, value_col) if len(df) > 10 else df
    
    # 创建饼图
    plt.pie(plot_df[value_col], labels=plot_df[label_col], autopct='%1.1f%%', startangle=90)
    
    # 设置标题
    plt.title(f'{value_col}占比分布', fontsize=16)
    
    # 确保饼图是圆形
    plt.axis('equal')

plt.tight_layout()
"""
# 地图
@staticmethod
def map_chart_template():
    return """
# 地图呈现
from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.globals import ChartType

# 识别地理维度列
geo_cols = ['省份', '地市', '区县', '乡镇', '村区']
geo_col = None
map_type = 'china'

# 找到存在的地理维度列
for col in geo_cols:
    if col in df.columns:
        geo_col = col
        if col == '省份':
            map_type = 'china'
        elif col == '地市':
            # 需要根据具体省份设置，这里以广东省为例
            map_type = '广东'
        break

# 找到数值列（如基站数量、小区数量等）
num_cols = df.select_dtypes(include=['number']).columns

if geo_col and len(num_cols) > 0:
    value_col = num_cols[0]
    
    # 聚合数据
    map_data = df.groupby(geo_col)[value_col].sum().reset_index()
    
    # 准备地图数据
    data_list = [(row[geo_col], float(row[value_col])) for _, row in map_data.iterrows()]
    
    # 创建地图
    map_chart = (
        Map(init_opts=opts.InitOpts(width="1200px", height="800px"))
        .add(
            series_name=value_col,
            data_pair=data_list,
            maptype=map_type,
            is_map_symbol_show=False,
        )
        .set_series_opts(
            label_opts=opts.LabelOpts(is_show=True, formatter="{b}: {c}")
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"{geo_col}{value_col}分布图",
                subtitle=f"数据范围: {map_data[value_col].min():.0f} - {map_data[value_col].max():.0f}",
                pos_left="center"
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="item",
                formatter="{b}<br/>{a}: {c}"
            ),
            visualmap_opts=opts.VisualMapOpts(
                is_calculable=True,
                dimension=0,
                pos_left="10",
                pos_bottom="10",
                min_=float(map_data[value_col].min()),
                max_=float(map_data[value_col].max()),
                range_text=["高", "低"],
                text_gap=10,
                orient="horizontal",
                is_piecewise=True,
                pieces=[
                    {"min": map_data[value_col].quantile(0.8), "label": "高", "color": "#d94e5d"},
                    {"min": map_data[value_col].quantile(0.6), "max": map_data[value_col].quantile(0.8), 
                     "label": "较高", "color": "#f57c7c"},
                    {"min": map_data[value_col].quantile(0.4), "max": map_data[value_col].quantile(0.6), 
                     "label": "中等", "color": "#ffb980"},
                    {"min": map_data[value_col].quantile(0.2), "max": map_data[value_col].quantile(0.4), 
                     "label": "较低", "color": "#ffd9a6"},
                    {"max": map_data[value_col].quantile(0.2), "label": "低", "color": "#fff2cc"}
                ]
            ),
            legend_opts=opts.LegendOpts(is_show=True, pos_top="5%", pos_right="5%")
        )
    )
    
    # 渲染地图
    # 在Jupyter Notebook中显示
    # map_chart.render_notebook()
    
    # 或保存为HTML文件
    map_chart.render("map_visualization.html")
    print("地图已保存为: map_visualization.html")
    
    # 同时显示数据统计
    print(f"\\n{geo_col}统计信息:")
    print(f"总{value_col}: {map_data[value_col].sum():.0f}")
    print(f"平均{value_col}: {map_data[value_col].mean():.2f}")
    print(f"\\nTop 10 {geo_col}:")
    print(map_data.nlargest(10, value_col).to_string(index=False))
"""

@staticmethod
def heatmap_template():
    return """
# 热力地图（适用于更细粒度的地理数据）
from pyecharts import options as opts
from pyecharts.charts import Geo
from pyecharts.globals import ChartType, SymbolType

# 假设数据中有经纬度信息
if '经度' in df.columns and '纬度' in df.columns:
    # 找到数值列
    num_cols = df.select_dtypes(include=['number']).columns
    value_cols = [col for col in num_cols if col not in ['经度', '纬度']]
    
    if value_cols:
        value_col = value_cols[0]
        
        # 准备热力图数据
        geo_data = []
        for _, row in df.iterrows():
            if pd.notna(row['经度']) and pd.notna(row['纬度']):
                # 如果有地点名称
                name = row.get('名称', f"点_{_}")
                geo_data.append((name, float(row[value_col])))
        
        # 创建地理坐标图
        geo_chart = (
            Geo(init_opts=opts.InitOpts(width="1200px", height="800px"))
            .add_schema(
                maptype="china",
                itemstyle_opts=opts.ItemStyleOpts(color="#323c48", border_color="#111"),
                emphasis_itemstyle_opts=opts.ItemStyleOpts(color="#2a333d")
            )
            .add(
                series_name=value_col,
                data_pair=geo_data,
                type_=ChartType.HEATMAP,
                symbol_size=12,
                effect_opts=opts.EffectOpts(is_show=True),
            )
            .set_series_opts(
                label_opts=opts.LabelOpts(is_show=False)
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"{value_col}热力分布图",
                    pos_left="center"
                ),
                visualmap_opts=opts.VisualMapOpts(
                    is_calculable=True,
                    min_=float(df[value_col].min()),
                    max_=float(df[value_col].max()),
                    range_text=["高", "低"],
                    pos_left="10",
                    pos_bottom="10"
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="item",
                    formatter="{b}: {c}"
                )
            )
        )
        
        # 渲染热力图
        geo_chart.render("heatmap_visualization.html")
        print("热力图已保存为: heatmap_visualization.html")
"""
