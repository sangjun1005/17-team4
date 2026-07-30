import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ==========================================
# 1. 한글 폰트 및 그래픽 기본 설정
# ==========================================
def setup_korean_font():
    """운영체제별 한글 폰트 자동 설정"""
    system_name = plt.os.name if hasattr(plt, 'os') else os.name
    
    # 시스템 설치 폰트 탐색
    font_names = [f.name for f in fm.fontManager.ttflist]
    
    if 'Malgun Gothic' in font_names:
        plt.rc('font', family='Malgun Gothic')  # Windows
    elif 'AppleGothic' in font_names:
        plt.rc('font', family='AppleGothic')   # Mac
    else:
        # Linux / Colab / Noto Sans 계열
        cjk_fonts = [f.fname for f in fm.fontManager.ttflist if 'cjk' in f.fname.lower() or 'nanum' in f.fname.lower()]
        if cjk_fonts:
            font_prop = fm.FontProperties(fname=cjk_fonts[0])
            plt.rc('font', family=font_prop.get_name())
        else:
            plt.rc('font', family='DejaVu Sans')
            
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

setup_korean_font()

# ==========================================
# 2. 데이터 로드 및 전처리
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'

# 개정된 결과 파일 우선 탐색, 없을 경우 기존 파일 사용
file_path = os.path.join(BASE_DIR, 'weather_weights_summary_2015_onwards.csv')

if not os.path.exists(file_path):
    print(f"[오류] {file_path} 요약 파일이 존재하지 않습니다.")
    exit()

df = pd.read_csv(file_path)

# $R^2$ 컬럼명 유연 처리 (Marginal_R2 또는 Pseudo_R2)
r2_col = 'Marginal_R2' if 'Marginal_R2' in df.columns else 'Pseudo_R2'

# 라벨 생성 (예: 급성 기관지염 (J20))
df['Disease_Label'] = df['질병명'] + ' (' + df['질병코드'] + ')'

print(f"로드된 데이터: 총 {len(df)}개 질병 분석 결과")

# Output 저장 폴더
output_dir = os.path.join(BASE_DIR, 'presentation_charts')
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 시각화 1: 질병-기상 요인 가중치 히트맵 (Heatmap)
# ==========================================
plt.figure(figsize=(10, 6.5))
heatmap_df = df.set_index('Disease_Label')[['기온_가중치(%)', 'PM25_가중치(%)', '습도_가중치(%)', '일교차_가중치(%)']]

ax1 = sns.heatmap(
    heatmap_df, 
    annot=True, 
    fmt=".1f", 
    cmap="YlOrRd", 
    cbar_kws={'label': '가중치 비중 (%)'},
    linewidths=0.8,
    linecolor='white'
)

plt.title('질병 - 기상/대기 요인별 감수성 가중치 히트맵', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('기상 및 대기 요인', fontsize=11, fontweight='bold', labelpad=10)
plt.ylabel('질병명 (질병코드)', fontsize=11, fontweight='bold', labelpad=10)
ax1.set_xticklabels(['기온', 'PM2.5 (미세먼지)', '습도', '일교차'], rotation=0)

plt.tight_layout()
path1 = os.path.join(output_dir, '1_weight_heatmap.png')
plt.savefig(path1, dpi=300)
plt.close()
print(f"[1/3] 히트맵 저장 완료: {path1}")

# ==========================================
# 시각화 2: 누적 막대 + 모델 설명력(R2) 이중축 그래프
# ==========================================
fig, ax_bar = plt.subplots(figsize=(12, 7))

# 설명력($R^2$) 기준 오름차순 정렬
df_sorted = df.sort_values(by=r2_col, ascending=True).reset_index(drop=True)
y_pos = np.arange(len(df_sorted))

temp = df_sorted['기온_가중치(%)']
pm25 = df_sorted['PM25_가중치(%)']
humid = df_sorted['습도_가중치(%)']
dtr = df_sorted['일교차_가중치(%)']

# 누적 막대 그래프
ax_bar.barh(y_pos, temp, label='기온', color='#FF6B6B', alpha=0.9)
ax_bar.barh(y_pos, pm25, left=temp, label='PM2.5 (미세먼지)', color='#4ECDC4', alpha=0.9)
ax_bar.barh(y_pos, humid, left=temp+pm25, label='습도', color='#45B7D1', alpha=0.9)
ax_bar.barh(y_pos, dtr, left=temp+pm25+humid, label='일교차', color='#FFA07A', alpha=0.9)

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(df_sorted['Disease_Label'], fontsize=10)
ax_bar.set_xlabel('기상 변수 가중치 구성비 (%)', fontsize=11, fontweight='bold')
ax_bar.set_xlim(0, 100)

# 이중 축 (Secondary Axis) - R2 라인
ax_line = ax_bar.twiny()
ax_line.plot(df_sorted[r2_col], y_pos, color='#2C3E50', marker='o', markersize=7, linewidth=2.5, label=f'모델 설명력 ({r2_col})')
ax_line.set_xlabel(f'모델 설명력 ({r2_col})', fontsize=11, fontweight='bold', color='#2C3E50')
ax_line.set_xlim(0, max(1.0, df_sorted[r2_col].max() * 1.15))

# 범례 통합
lines_b, labels_b = ax_bar.get_legend_handles_labels()
lines_l, labels_l = ax_line.get_legend_handles_labels()
ax_bar.legend(lines_b + lines_l, labels_b + labels_l, loc='lower right', bbox_to_anchor=(1.0, 1.02), ncol=3)

plt.title('질병별 기상 가중치 구성비 및 모델 설명력($R^2$) 통합 분석', fontsize=14, fontweight='bold', pad=25)
plt.tight_layout()
path2 = os.path.join(output_dir, '2_stacked_bar_with_r2.png')
plt.savefig(path2, dpi=300)
plt.close()
print(f"[2/3] 이중축 누적 막대 그래프 저장 완료: {path2}")

# ==========================================
# 시각화 3: 질병군(카테고리)별 기상 변수 평균 영향력
# ==========================================
def categorize_disease(code):
    if code.startswith('J'):
        return '호흡기 및 알레르기 질환'
    elif code.startswith('M') or code.startswith('I'):
        return '근골격계 / 순환기 질환'
    else:
        return '기타 소화기 / 피부 질환'

df['질병군'] = df['질병코드'].apply(categorize_disease)
cat_avg = df.groupby('질병군')[['기온_가중치(%)', 'PM25_가중치(%)', '습도_가중치(%)', '일교차_가중치(%)']].mean().reset_index()

fig, ax3 = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(cat_avg))
width = 0.2

b1 = ax3.bar(x - 1.5*width, cat_avg['기온_가중치(%)'], width, label='기온', color='#FF6B6B')
b2 = ax3.bar(x - 0.5*width, cat_avg['PM25_가중치(%)'], width, label='PM2.5 (미세먼지)', color='#4ECDC4')
b3 = ax3.bar(x + 0.5*width, cat_avg['습도_가중치(%)'], width, label='습도', color='#45B7D1')
b4 = ax3.bar(x + 1.5*width, cat_avg['일교차_가중치(%)'], width, label='일교차', color='#FFA07A')

ax3.set_ylabel('평균 가중치 비중 (%)', fontsize=11, fontweight='bold')
ax3.set_title('질병군(카테고리)별 기상 변수 평균 영향력 비교', fontsize=14, fontweight='bold', pad=15)
ax3.set_xticks(x)
ax3.set_xticklabels(cat_avg['질병군'], fontsize=11, fontweight='bold')
ax3.legend(loc='upper right')

# 막대 상단 수치 표기
for bars in [b1, b2, b3, b4]:
    for bar in bars:
        height = bar.get_height()
        if height > 1.5:
            ax3.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5)

plt.tight_layout()
path3 = os.path.join(output_dir, '3_category_comparison.png')
plt.savefig(path3, dpi=300)
plt.close()
print(f"[3/3] 질병군 카테고리 비교 차트 저장 완료: {path3}")

print("\n모든 발표용 고해상도 시각화 자료가 'presentation_charts' 폴더에 성공적으로 생성되었습니다!")