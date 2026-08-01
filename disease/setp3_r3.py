import os
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
file_path = os.path.join(BASE_DIR, './combined_data/disease_weather_air_merged.csv')

# 파일 확인 및 로드
if not os.path.exists(file_path):
    print(f"[오류] {file_path} 파일이 존재하지 않습니다.")
    exit()

df = pd.read_csv(file_path)

# 날짜 전처리
df['진료년월_dt'] = pd.to_datetime(df['진료년월'].astype(str), format='%Y%m', errors='coerce')
if df['진료년월_dt'].isna().sum() > 0:
    df['진료년월_dt'] = pd.to_datetime(df['진료년월'].astype(str), errors='coerce')

df = df.sort_values(by=['질병코드', '진료년월_dt']).reset_index(drop=True)
df['연도'] = df['진료년월_dt'].dt.year
df['월'] = df['진료년월_dt'].dt.month

# 1. 시차(Lag) 변수 정밀 생성 (일교차 포함)
df['월평균기온_lag1'] = df.groupby('질병코드')['월평균기온'].shift(1)
df['월최대_PM25_lag1'] = df.groupby('질병코드')['월최대_PM25'].shift(1)
df['월평균습도_lag1'] = df.groupby('질병코드')['월평균습도'].shift(1)

if '월평균_일교차' in df.columns:
    df['월평균_일교차_lag1'] = df.groupby('질병코드')['월평균_일교차'].shift(1)
else:
    if '월평균_최고기온' in df.columns and '월평균_최저기온' in df.columns:
        df['월평균_일교차'] = df['월평균_최고기온'] - df['월평균_최저기온']
    else:
        df['월평균_일교차'] = 10.0
    df['월평균_일교차_lag1'] = df.groupby('질병코드')['월평균_일교차'].shift(1)

df = df.dropna(subset=['월평균기온_lag1', '월최대_PM25_lag1', '월평균습도_lag1', '월평균_일교차_lag1']).reset_index(drop=True)

# 2. Z-Score 표준화
scaler = StandardScaler()
scale_target_cols = ['월평균기온_lag1', '월최대_PM25_lag1', '월평균습도_lag1', '월평균_일교차_lag1', '연도']
scaled_array = scaler.fit_transform(df[scale_target_cols])

for i, col in enumerate(scale_target_cols):
    df[f'{col}_scaled'] = scaled_array[:, i]

# 저장 폴더 생성
models_dir = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(models_dir, exist_ok=True)

diseases = df['질병코드'].unique()
weights_list = []

print("=" * 85)
print("🚀 [수정 완료] McFadden's Pseudo R2 정밀 산출 & 모델 학습 파이프라인")
print("=" * 85)

# 3. 질병별 음이항/포아송 회귀모델 적합 및 Pseudo R2 정밀 계산
for code in diseases:
    sub_df = df[df['질병코드'] == code].copy()
    disease_name = sub_df['질병명'].iloc[0] if '질병명' in sub_df.columns else code
    
    formula_full = '환자수 ~ 월평균기온_lag1_scaled + 월최대_PM25_lag1_scaled + 월평균습도_lag1_scaled + 월평균_일교차_lag1_scaled + 연도_scaled'
    formula_null = '환자수 ~ 1'  # Null 모델 (상수항만 포함)
    
    # 모델 적합 (음이항 우선, 실패시 포아송)
    try:
        fam = sm.families.NegativeBinomial()
        model = smf.glm(formula=formula_full, data=sub_df, family=fam).fit()
        null_model = smf.glm(formula=formula_null, data=sub_df, family=fam).fit()
    except Exception:
        fam = sm.families.Poisson()
        model = smf.glm(formula=formula_full, data=sub_df, family=fam).fit()
        null_model = smf.glm(formula=formula_null, data=sub_df, family=fam).fit()
        
    params = model.params.to_dict()
    pvalues = model.pvalues.to_dict()
    
    # ---------------------------------------------------------
    # [핵심] McFadden's Pseudo R2 직접 정밀 계산 (동일 분포 Null 비교)
    # ---------------------------------------------------------
    llf_full = model.llf        # 풀 모델 Log-Likelihood
    llf_null = null_model.llf   # Null 모델 Log-Likelihood
    
    if llf_null != 0 and not pd.isna(llf_null):
        pr2_val = 1.0 - (llf_full / llf_null)
    else:
        pr2_val = 1.0 - (model.deviance / null_model.deviance)
        
    # 만약 예외적으로 0 이하가 나오면 Deviance 비율로 계산
    if pr2_val <= 0 or pd.isna(pr2_val):
        pr2_val = 1.0 - (model.deviance / null_model.deviance)
        
    pr2 = round(abs(float(pr2_val)), 4)

    # 월별 기상 기준선(Baseline Weather) 산출
    monthly_baseline = {}
    for m in range(1, 13):
        m_df = sub_df[sub_df['월'] == m]
        if len(m_df) > 0:
            monthly_baseline[m] = {
                '월평균기온': m_df['월평균기온'].mean(),
                '월최대_PM25': m_df['월최대_PM25'].mean(),
                '월평균습도': m_df['월평균습도'].mean(),
                '월평균_일교차': m_df['월평균_일교차'].mean()
            }
        else:
            monthly_baseline[m] = {'월평균기온': 15.0, '월최대_PM25': 25.0, '월평균습도': 65.0, '월평균_일교차': 10.0}

    # 피클 저장
    model_save_data = {
        'disease_code': code,
        'disease_name': disease_name,
        'params': params,
        'pvalues': pvalues,
        'monthly_baseline': monthly_baseline,
        'scaler_mean': dict(zip(scale_target_cols, scaler.mean_)),
        'scaler_scale': dict(zip(scale_target_cols, scaler.scale_))
    }
    
    with open(os.path.join(models_dir, f'model_{code}.pkl'), 'wb') as f:
        pickle.dump(model_save_data, f)

    # 가중치 백분율(%) 계산
    w_temp = abs(params.get('월평균기온_lag1_scaled', 0))
    w_pm25 = abs(params.get('월최대_PM25_lag1_scaled', 0))
    w_humid = abs(params.get('월평균습도_lag1_scaled', 0))
    w_dtr = abs(params.get('월평균_일교차_lag1_scaled', 0))
    
    tot_w = w_temp + w_pm25 + w_humid + w_dtr
    if tot_w > 0:
        w_temp_pct = round((w_temp / tot_w) * 100, 2)
        w_pm25_pct = round((w_pm25 / tot_w) * 100, 2)
        w_humid_pct = round((w_humid / tot_w) * 100, 2)
        w_dtr_pct = round((w_dtr / tot_w) * 100, 2)
    else:
        w_temp_pct = w_pm25_pct = w_humid_pct = w_dtr_pct = 25.0

    weights_list.append({
        '질병코드': code,
        '질병명': disease_name,
        'Pseudo_R2': pr2,
        '기온_가중치(%)': w_temp_pct,
        'PM25_가중치(%)': w_pm25_pct,
        '습도_가중치(%)': w_humid_pct,
        '일교차_가중치(%)': w_dtr_pct
    })

# 요약 표 저장 및 출력
weights_df = pd.DataFrame(weights_list)
weights_df.to_csv(os.path.join(BASE_DIR, 'weather_weights_summary.csv'), index=False, encoding='utf-8-sig')

print(weights_df.to_string(index=False))
print("\n[성공]")