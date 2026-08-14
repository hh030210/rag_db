import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 数据准备 ----------------------
# 指标名称
categories = ['Precision', 'Recall', 'F1-Score', '1-Retention']
N = len(categories)

# 方法名称与对应数据（Retention 转换为 1-Retention，雷达图要求数值越大越好）
methods = {
    'Rule': [0.82, 0.56, 0.666, 1 - 0.44],
    'LLM-Only': [0.84, 0.79, 0.814, 1 - 0.21],
    'LLM-Denoise': [0.89, 0.89, 0.890, 1 - 0.11],
    'Ours(Reorg)': [0.91, 0.92, 0.915, 1 - 0.08]
}

# 配色（与原图一致，高对比度）
colors = {
    'Rule': '#1f77b4',        # 蓝色
    'LLM-Only': '#ff7f0e',    # 橙色
    'LLM-Denoise': '#2ca02c', # 绿色
    'Ours(Reorg)': '#d62728'  # 红色
}

# 雷达图角度计算（闭合图形）
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# ---------------------- 绘图设置 ----------------------
plt.figure(figsize=(8, 8), dpi=150)
ax = plt.subplot(111, polar=True)

# 绘制每个方法的雷达线 + 填充
for method, values in methods.items():
    values += values[:1]  # 闭合图形
    ax.plot(angles, values, color=colors[method], linewidth=2, label=method)
    ax.fill(angles, values, color=colors[method], alpha=0.15)

# ---------------------- 图表美化 ----------------------
# 轴标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)

# 径向轴（0-1 范围，与原图一致）
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=8, color='gray')

# 标题
plt.title('Comprehensive Metric Radar Map (Artificial Noise)', fontsize=13, pad=30)

# 图例（避免遮挡）
ax.legend(loc='lower left', bbox_to_anchor=(-0.2, -0.2), fontsize=9)

# 自动适配布局，防止标签截断
plt.tight_layout()

# 保存300dpi高清图（直接用于论文）
plt.savefig('radar_map_high_res.png', dpi=300, bbox_inches='tight')
plt.show()