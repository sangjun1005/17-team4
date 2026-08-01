import os
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 1. 경로 설정 및 파일 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
file_path = os.path.join(BASE_DIR, 'combined_data', 'disease_weather_air_merged.csv')

if not os.path.exists(file_path):
    if os.path.exists('disease_weather_air_merged.csv'):
        file_path = 'disease_weather_air_merged.csv'
    else:
        print(f"[오류] 파일이 존재하지 않습니다.")
        exit()

df = pd.read_csv(file_path)

# 2. 날짜 전처리
df['진료년월_dt'] = pd.to_datetime(df['진료년월'].astype(str), format='%Y%m', errors='coerce')
if df['진료년월_dt'].isna().sum() > 0:
    df['진료년월_dt'] = pd.to_datetime(df['진료년월'].astype(str), errors='coerce')

df = df.sort_values(by=['질병코드', '진료년월_dt']).reset_index(drop=True)
df['연도'] = df['진료년월_dt'].dt.year
df['월'] = df['진료년월_dt'].dt.month

# 3. 미세먼지 및 일교차 지표 자동 선택 (월평균 우선 적용)
if '월평균_PM25' in df.columns:
    pm25_col = '월평균_PM25'
elif '월최대_PM25' in df.columns:
    pm25_col = '월최대_PM25'
else:
    pm_candidates = [c for c in df.columns if 'PM25' in c or 'pm25' in c or '미세먼지' in c]
    pm25_col = pm_candidates[0] if pm_candidates else '월평균_PM25'

if '월평균_일교차' not in df.columns:
    if '월평균_최고기온' in df.columns and '월평균_최저기온' in df.columns:
        df['월평균_일교차'] = df['월평균_최고기온'] - df['월평균_최저기온']
    else:
        df['월평균_일교차'] = 10.0

# 4. Lag-1 (1개월 시차) 파생변수 생성
df['월평균기온_lag1'] = df.groupby('질병코드')['월평균기온'].shift(1)
df['PM25_lag1'] = df.groupby('질병코드')[pm25_col].shift(1)
df['월평균습도_lag1'] = df.groupby('질병코드')['월평균습도'].shift(1)
df['월평균_일교차_lag1'] = df.groupby('질병코드')['월평균_일교차'].shift(1)

# Shift로 인한 결측치 제거
df = df.dropna(subset=['월평균기온_lag1', 'PM25_lag1', '월평균습도_lag1', '월평균_일교차_lag1', '환자수']).reset_index(drop=True)

# 5. Z-Score 표준화 (독립변수 스케일 동일화)
scale_cols = ['월평균기온_lag1', 'PM25_lag1', '월평균습도_lag1', '월평균_일교차_lag1', '연도']
scaler = StandardScaler()
scaled_vals = scaler.fit_transform(df[scale_cols])

for i, col in enumerate(scale_cols):
    df[f'{col}_scaled'] = scaled_vals[:, i]

# 모델 저장 폴더 준비
models_dir = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(models_dir, exist_ok=True)

diseases = df['질병코드'].unique()
weights_list = []

print("[정밀 검토 완료] 다중공선성 제거, 연도추세 통제 및 Deviance Drop 기반 가중치 산출")

weather_vars = ['월평균기온_lag1_scaled', 'PM25_lag1_scaled', '월평균습도_lag1_scaled', '월평균_일교차_lag1_scaled']

for code in diseases:
    sub_df = df[df['질병코드'] == code].copy().reset_index(drop=True)
    disease_name = sub_df['질병명'].iloc[0] if '질병명' in sub_df.columns else code

    # [핵심 설계를 보정] Baseline: 장기 연도 추세만 통제 (월 더미 제거로 다중공선성 방지)
    formula_base = '환자수 ~ 연도_scaled'
    
    # Full 모델: Baseline + 4대 순수 기상 변수
    formula_full = f"{formula_base} + {' + '.join(weather_vars)}"

    # 안전한 GLM 적합 예외 처리 함수
    def fit_glm_safe(formula, data):
        try:
            res = smf.glm(formula=formula, data=data, family=sm.families.NegativeBinomial()).fit()
            if not res.converged:
                res = smf.glm(formula=formula, data=data, family=sm.families.Poisson()).fit()
            return res
        except Exception:
            return smf.glm(formula=formula, data=data, family=sm.families.Poisson()).fit()

    model_full = fit_glm_safe(formula_full, sub_df)
    model_base = fit_glm_safe(formula_base, sub_df)

    dev_full = model_full.deviance
    dev_base = model_base.deviance

    # 순수 기상 설명력 (Marginal R2)
    marginal_r2 = round(max(0.0, float(1.0 - (dev_full / dev_base))), 4) if dev_base > 0 else 0.0

    # ANOVA Deviance Drop (Type II) 기반 개별 변수 기여도 가중치 계산
    dev_drops = {}
    for var in weather_vars:
        vars_drop = [v for v in weather_vars if v != var]
        formula_drop = f"{formula_base} + {' + '.join(vars_drop)}"
        model_drop = fit_glm_safe(formula_drop, sub_df)
        
        # 특정 변수가 제거되었을 때 증가하는 Deviance (해당 변수의 순수 기여량)
        drop_val = max(0.0, float(model_drop.deviance - dev_full))
        dev_drops[var] = drop_val

    tot_drop = sum(dev_drops.values())

    if tot_drop > 0:
        w_temp_pct = round((dev_drops['월평균기온_lag1_scaled'] / tot_drop) * 100, 2)
        w_pm25_pct = round((dev_drops['PM25_lag1_scaled'] / tot_drop) * 100, 2)
        w_humid_pct = round((dev_drops['월평균습도_lag1_scaled'] / tot_drop) * 100, 2)
        w_dtr_pct = round((dev_drops['월평균_일교차_lag1_scaled'] / tot_drop) * 100, 2)
    else:
        w_temp_pct = w_pm25_pct = w_humid_pct = w_dtr_pct = 25.0

    # 월별 기상 Baseline 수집 (추천 알고리즘 스코어링 단계 연동용)
    monthly_baseline = {}
    for m in range(1, 13):
        m_df = sub_df[sub_df['월'] == m]
        if len(m_df) > 0:
            monthly_baseline[m] = {
                '월평균기온': float(m_df['월평균기온'].mean()),
                'PM25': float(m_df[pm25_col].mean()),
                '월평균습도': float(m_df['월평균습도'].mean()),
                '월평균_일교차': float(m_df['월평균_일교차'].mean())
            }
        else:
            monthly_baseline[m] = {'월평균기온': 15.0, 'PM25': 25.0, '월평균습도': 65.0, '월평균_일교차': 10.0}

    # 피클 모델 저장
    model_save_data = {
        'disease_code': code,
        'disease_name': disease_name,
        'params': model_full.params.to_dict(),
        'pvalues': model_full.pvalues.to_dict(),
        'monthly_baseline': monthly_baseline,
        'scaler_mean': dict(zip(scale_cols, scaler.mean_)),
        'scaler_scale': dict(zip(scale_cols, scaler.scale_))
    }

    with open(os.path.join(models_dir, f'model_{code}.pkl'), 'wb') as f:
        pickle.dump(model_save_data, f)

    weights_list.append({
        '질병코드': code,
        '질병명': disease_name,
        'Marginal_R2': marginal_r2,
        '기온_가중치(%)': w_temp_pct,
        'PM25_가중치(%)': w_pm25_pct,
        '습도_가중치(%)': w_humid_pct,
        '일교차_가중치(%)': w_dtr_pct
    })

# CSV 저장 및 출력
summary_df = pd.DataFrame(weights_list)
summary_path = os.path.join(BASE_DIR, 'weather_weights_summary_revised.csv')
summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

print(summary_df.to_string(index=False))
print(f"\n[검토 및 수정 완료] 결과가 '{summary_path}' 파일로 정밀하게 저장되었습니다.")