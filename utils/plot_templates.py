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
