import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

font_names = [f.name for f in fm.fontManager.ttflist]
if 'Malgun Gothic' in font_names:
  plt.rc('font', family='Malgun Gothic')
else:
  cjk_fonts = [
      f.fname
      for f in fm.fontManager.ttflist
      if 'cjk' in f.fname.lower() or 'nanum' in f.fname.lower()
  ]
  if cjk_fonts:
    font_prop = fm.FontProperties(fname=cjk_fonts[0])
    plt.rc('font', family=font_prop.get_name())
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('./disease/combined_data/disease_weather_air_merged.csv')
target_codes = ['I10', 'J06', 'J20', 'J30', 'J45', 'K29', 'L23', 'M17']
df_sub = df[df['질병코드'].isin(target_codes)]

stats = (
    df_sub.groupby(['질병코드', '질병명'])['환자수']
    .agg(['mean', 'var'])
    .reset_index()
)
stats['ratio'] = stats['var'] / stats['mean']
stats['질병표시'] = stats['질병명'] + ' (' + stats['질병코드'] + ')'
stats = stats.sort_values('ratio', ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sns.scatterplot(
    data=stats, x='mean', y='var', s=200, color='#E74C3C', ax=ax1, zorder=5
)

for i, row in stats.iterrows():
  ax1.text(
      row['mean'] * 1.05,
      row['var'] * 1.05,
      row['질병코드'],
      fontsize=11,
      fontweight='bold',
  )

min_val = min(stats['mean'].min(), stats['var'].min()) * 0.5
max_val = max(stats['mean'].max(), stats['var'].max()) * 2
x_line = np.logspace(np.log10(min_val), np.log10(max_val), 100)
ax1.plot(x_line, x_line, 'k--', label='포아송 가정선 (분산 = 평균)', linewidth=2)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title(
    '질병별 환자 수 평균 vs 분산',
    fontsize=13,
    fontweight='bold',
    pad=12,
)
ax1.set_xlabel('환자 수 평균 (Mean, $\mu$)', fontsize=11, fontweight='bold')
ax1.set_ylabel('환자 수 분산 (Variance, $\sigma^2$)', fontsize=11, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, which='both', ls='--', alpha=0.5)

bars = ax2.barh(
    stats['질병표시'],
    stats['ratio'],
    color='#3498DB',
    edgecolor='black',
    height=0.6,
)
ax2.set_xscale('log')
ax2.set_title(
    '분산 / 평균',
    fontsize=13,
    fontweight='bold',
    pad=12,
)
ax2.set_xlabel(
    '과산포 비율 ($\sigma^2 / \mu$, 로그 스케일)', fontsize=11, fontweight='bold'
)
ax2.grid(True, which='both', ls='--', alpha=0.5)

for bar in bars:
  width = bar.get_width()
  ax2.text(
      width * 1.1,
      bar.get_y() + bar.get_height() / 2,
      f'{width:,.0f}배',
      va='center',
      ha='left',
      fontsize=10,
      fontweight='bold',
      color='#2C3E50',
  )

plt.tight_layout()
plt.savefig('overdispersion_proof.png', dpi=300, bbox_inches='tight')
plt.show()