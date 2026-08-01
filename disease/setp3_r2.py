import os
import pickle
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# ---------------------------------------------------------
# 0. 한글 폰트 및 그래픽 설정
# ---------------------------------------------------------
sns.set_theme(style="whitegrid")
system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 1. 테스트용 30일 일별 기상 예보 데이터 (기온, 일교차, 미세먼지, 습도)
# ---------------------------------------------------------
np.random.seed(42)
dates = pd.date_range(start='2026-08-01', periods=30, freq='D')
df_val = pd.DataFrame({'날짜': dates})

# 기상 변동 생성 (평년 8월 기준)
df_val['월평균기온'] = np.random.normal(27.0, 3.0, 30)       # 기온
df_val['일교차'] = np.random.normal(8.0, 2.5, 30)            # 일교차
df_val['월최대_PM25'] = np.random.exponential(20, 30) + 10    # 미세먼지
df_val['월평균습도'] = np.random.normal(70.0, 10.0, 30)      # 습도

# ---------------------------------------------------------
# 2. [비교군 A] 기상청/건보공단 보건기상지수 가상 산출 공식 (KMA/NHIS 기준)
# ---------------------------------------------------------
# 기상청 천식지수: (기온 편차 음수) + (일교차) + (미세먼지) + (저습도)
z_temp_kma = (27.0 - df_val['월평균기온']) / 3.0
z_pm25_kma = (df_val['월최대_PM25'] - 18.0) / 12.0
z_range_kma = (df_val['일교차'] - 8.0) / 2.5

# 기상청 공식 근사치 (천식)
kma_asthma_raw = 50.0 + (z_temp_kma * 15.0) + (z_pm25_kma * 18.0) + (z_range_kma * 10.0)
df_val['기상청_천식지수'] = np.clip(np.round(kma_asthma_raw, 1), 0.0, 100.0)

# 기상청 공식 근사치 (심뇌혈관)
kma_cardio_raw = 50.0 + (z_temp_kma * 25.0) + (z_range_kma * 15.0)
df_val['기상청_심뇌혈관지수'] = np.clip(np.round(kma_cardio_raw, 1), 0.0, 100.0)

# ---------------------------------------------------------
# 3. [비교군 B] 우리 팀 회귀분석 기반 보건지수 산출 (Z-Score 방식)
# ---------------------------------------------------------
# 우리 모델 천식(J45) 회귀계수 적용 (예: b_temp = -0.15, b_pm25 = 0.22)
our_asthma_delta = ((df_val['월평균기온'] - 27.0)/3.0 * -0.18) + ((df_val['월최대_PM25'] - 18.0)/12.0 * 0.25)
our_asthma_score = 50.0 + (our_asthma_delta * 40.0)
df_val['우리모델_천식(J45)'] = np.clip(np.round(our_asthma_score, 1), 0.0, 100.0)

# 우리 모델 고혈압/심혈관(I10) 회귀계수 적용 (예: b_temp = -0.28)
our_cardio_delta = ((df_val['월평균기온'] - 27.0)/3.0 * -0.30) + ((df_val['월최대_PM25'] - 18.0)/12.0 * 0.08)
our_cardio_score = 50.0 + (our_cardio_delta * 40.0)
df_val['우리모델_고혈압(I10)'] = np.clip(np.round(our_cardio_score, 1), 0.0, 100.0)

# ---------------------------------------------------------
# 4. 정량적 검증 분석 (상관계수 및 등급 일치율)
# ---------------------------------------------------------
def get_kma_grade(score):
    if score >= 80: return '매우높음'
    elif score >= 65: return '높음'
    elif score >= 35: return '보통'
    else: return '낮음'

# 등급 부여
df_val['기상청_천식_등급'] = df_val['기상청_천식지수'].apply(get_kma_grade)
df_val['우리모델_천식_등급'] = df_val['우리모델_천식(J45)'].apply(get_kma_grade)

df_val['기상청_심혈관_등급'] = df_val['기상청_심뇌혈관지수'].apply(get_kma_grade)
df_val['우리모델_심혈관_등급'] = df_val['우리모델_고혈압(I10)'].apply(get_kma_grade)

# 지표 계산 - 천식
corr_asthma, p_asthma = pearsonr(df_val['기상청_천식지수'], df_val['우리모델_천식(J45)'])
acc_asthma = (df_val['기상청_천식_등급'] == df_val['우리모델_천식_등급']).mean() * 100
mae_asthma = np.abs(df_val['기상청_천식지수'] - df_val['우리모델_천식(J45)']).mean()

# 지표 계산 - 심뇌혈관
corr_cardio, p_cardio = pearsonr(df_val['기상청_심뇌혈관지수'], df_val['우리모델_고혈압(I10)'])
acc_cardio = (df_val['기상청_심혈관_등급'] == df_val['우리모델_심혈관_등급']).mean() * 100
mae_cardio = np.abs(df_val['기상청_심뇌혈관지수'] - df_val['우리모델_고혈압(I10)']).mean()

print("=" * 90)
print(" 🔬 [검증 결과] 기상청 보건기상지수 vs 우리 팀 질병 위험지수 비교")
print("=" * 90)
print(f"1. 천식 질환 지수 (KMA 천식지수 vs 우리모델 J45)")
print(f"   - 피어슨 상관계수 (r) : {corr_asthma:.4f} (p-value: {p_asthma:.4e})")
print(f"   - 4단계 등급 일치율  : {acc_asthma:.1f}%")
print(f"   - 평균 절대 오차(MAE): {mae_asthma:.2f}점\n")

print(f"2. 심뇌혈관 질환 지수 (KMA 심뇌혈관지수 vs 우리모델 I10)")
print(f"   - 피어슨 상관계수 (r) : {corr_cardio:.4f} (p-value: {p_cardio:.4e})")
print(f"   - 4단계 등급 일치율  : {acc_cardio:.1f}%")
print(f"   - 평균 절대 오차(MAE): {mae_cardio:.2f}점")
print("=" * 90)

# ---------------------------------------------------------
# 5. 검증 결과 시각화 (비교 대시보드 4종)
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(' 기상청 보건기상지수 vs 우리 모델 위험지수 일치성 검증 대시보드', fontsize=18, fontweight='bold')

# Chart 1: 천식 지수 시길 추이 비교
axes[0, 0].plot(df_val['날짜'], df_val['기상청_천식지수'], label='기상청 천식지수', color='blue', linestyle='--', marker='o')
axes[0, 0].plot(df_val['날짜'], df_val['우리모델_천식(J45)'], label='우리모델 천식(J45)', color='red', linewidth=2, marker='s')
axes[0, 0].set_title(f'① 천식 지수 추이 비교 (상관계수 r = {corr_asthma:.2f})', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('위험 지수 (0~100)')
axes[0, 0].tick_params(axis='x', rotation=30)
axes[0, 0].legend()

# Chart 2: 천식 지수 산점도 및 회귀선 (Correlation Scatter)
sns.regplot(x='기상청_천식지수', y='우리모델_천식(J45)', data=df_val, ax=axes[0, 1], color='purple')
axes[0, 1].set_title('② 천식 지수 산점도 (일치성 상관관계)', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('기상청 천식지수')
axes[0, 1].set_ylabel('우리모델 천식(J45) 점수')

# Chart 3: 심뇌혈관 지수 시계열 추이 비교
axes[1, 0].plot(df_val['날짜'], df_val['기상청_심뇌혈관지수'], label='기상청 심뇌혈관지수', color='green', linestyle='--', marker='o')
axes[1, 0].plot(df_val['날짜'], df_val['우리모델_고혈압(I10)'], label='우리모델 고혈압(I10)', color='orange', linewidth=2, marker='s')
axes[1, 0].set_title(f'③ 심뇌혈관 지수 추이 비교 (상관계수 r = {corr_cardio:.2f})', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('위험 지수 (0~100)')
axes[1, 0].tick_params(axis='x', rotation=30)
axes[1, 0].legend()

# Chart 4: 심뇌혈관 지수 산점도
sns.regplot(x='기상청_심뇌혈관지수', y='우리모델_고혈압(I10)', data=df_val, ax=axes[1, 1], color='darkgreen')
axes[1, 1].set_title('④ 심뇌혈관 지수 산점도 (일치성 상관관계)', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('기상청 심뇌혈관지수')
axes[1, 1].set_ylabel('우리모델 고혈압(I10) 점수')

plt.tight_layout()
output_img = os.path.join('.', '기상청_생활보건지수_검증_리포트.png')
plt.savefig(output_img, dpi=300)
print(f"\n 검증 시각화 그래프가 저장되었습니다: {output_img}")
plt.show()