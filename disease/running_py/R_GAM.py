import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pygam import PoissonGAM, s, l
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

# 1. 파일 경로 자동 인식 및 데이터 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, 'disease_weather_air_merged.csv')

df = pd.read_csv(file_path)

# 2. 날짜 및 트렌드/지연 변수 전처리
df['진료년월_dt'] = pd.to_datetime(df['진료년월'].astype(str), format='%Y%m', errors='coerce')
if df['진료년월_dt'].isna().sum() > 0:
    df['진료년월_dt'] = pd.to_datetime(df['진료년월'].astype(str), errors='coerce')

df = df.sort_values(by=['질병코드', '진료년월_dt']).reset_index(drop=True)

df['연도'] = df['진료년월_dt'].dt.year
df['월'] = df['진료년월_dt'].dt.month

# [Point 1] 음이항 회귀 overflow 방지를 위한 스케일링 변수 생성
df['연도_scaled'] = df['연도'] - df['연도'].min()  # 시작 연도를 0으로 설정
df['환자수_lag12'] = df.groupby('질병코드')['환자수'].shift(12)
df['환자수_lag12_k'] = df['환자수_lag12'] / 1000.0  # 천 명 단위로 스케일 조정

weather_cols = ['월평균기온', '월평균최고기온', '월평균일교차', '월평균습도', '월평균기압', '월평균_PM10', '월최대_PM25']
for col in weather_cols:
    if col in df.columns:
        df[f'{col}_lag1'] = df[col].shift(1)

# 코로나19 기간(2020~2022) 제외
covid_mask = (df['진료년월_dt'] >= '2020-01-01') & (df['진료년월_dt'] <= '2022-12-31')
df_clean = df[~covid_mask].dropna().reset_index(drop=True)

# M17 무릎관절증 필터링
target_code = 'M17'
sub_df = df_clean[df_clean['질병코드'] == target_code].copy().reset_index(drop=True)

print(f"📌 [{target_code}] 스케일링 전처리 완료 (총 {len(sub_df)}개 월별 데이터)\n")

# ==========================================
# 3. [통계 모델 1] 음이항 회귀 (수치 폭발 보정)
# ==========================================
print("==================================================")
print("📊 [1] Statsmodels 음이항 회귀 (Negative Binomial)")
print("==================================================")

# 수치 스케일이 조정된 변수로 formula 구성
formula = "환자수 ~ 연도_scaled + C(월) + 환자수_lag12_k + 월평균기온 + 월평균기압_lag1 + 월최대_PM25_lag1"
nb_model = smf.negativebinomial(formula, data=sub_df).fit(maxiter=200, disp=False)

print(nb_model.summary())

# ==========================================
# 4. [통계 모델 2] pyGAM (파이썬 일반화 가법 모델)
# ==========================================
print("\n==================================================")
print("📈 [2] pyGAM 비선형 스플라인 회귀 (PyGAM)")
print("==================================================")

X_cols = ['환자수_lag12', '연도', '월평균기온', '월평균기압_lag1', '월최대_PM25_lag1']
X = sub_df[X_cols].values
y = sub_df['환자수'].values

gam = PoissonGAM(
    s(0, n_splines=5) +  # 환자수_lag12 (스플라인)
    l(1) +               # 연도 (선형 트렌드)
    s(2, n_splines=5) +  # 월평균기온
    s(3, n_splines=5) +  # 월평균기압_lag1
    s(4, n_splines=5)    # 월최대_PM25_lag1
).fit(X, y)

print(gam.summary())

# ==========================================
# 5. GAM 비선형 기상 반응 곡선 시각화
# ==========================================
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
titles = ['월평균기온 비선형 영향', '월평균기압(Lag1) 비선형 영향', '월최대 PM2.5(Lag1) 비선형 영향']
feature_indices = [2, 3, 4]

for i, (idx, title) in enumerate(zip(feature_indices, titles)):
    XX = gam.generate_X_grid(term=i+1)
    pdep, confi = gam.partial_dependence(term=i+1, X=XX, width=0.95)
    
    axs[i].plot(XX[:, idx], pdep, color='crimson', lw=2)
    axs[i].fill_between(XX[:, idx], confi[:, 0], confi[:, 1], color='crimson', alpha=0.15)
    axs[i].set_title(title, fontsize=12, fontweight='bold')
    axs[i].set_xlabel(X_cols[idx])
    axs[i].set_ylabel('환자수 변동 영향도')
    axs[i].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'gam_weather_response_curve.png'), dpi=300)
print("\n💡 [시각화 완료] 'gam_weather_response_curve.png' 그래프 저장 완료!")
plt.show()