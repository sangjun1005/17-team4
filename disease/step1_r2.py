import os
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
file_path = os.path.join(BASE_DIR, 'disease_weather_air_merged.csv')
models_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(models_dir, exist_ok=True)

if not os.path.exists(file_path):
    alt_path = os.path.join(BASE_DIR, './combined_data/disease_weather_air_merged.csv')
    if os.path.exists(alt_path):
        file_path = alt_path
    else:
        print(f"❌ [오류] {file_path} 또는 {alt_path} 파일이 존재하지 않습니다.")
        exit()

df = pd.read_csv(file_path)

# 2. 안전한 날짜 전처리 ('YYYY-MM' 및 'YYYYMM' 형식 모두 에러 없이 파싱)
df['진료년월_dt'] = pd.to_datetime(df['진료년월'].astype(str), errors='coerce')
df['연도'] = df['진료년월_dt'].dt.year
df['월'] = df['진료년월_dt'].dt.month

# 3. 푸리에 항(Seasonality Fourier Terms) 생성 (과적합 방지 및 자유도 확보)
df['sin1'] = np.sin(2 * np.pi * df['월'] / 12)
df['cos1'] = np.cos(2 * np.pi * df['월'] / 12)

# 4. 필터링 (2015년 이후, 코로나 기간 2020~2022 제외)
df_filtered = df[(df['연도'] >= 2015) & (~df['연도'].between(2020, 2022))].copy()

scale_target_cols = ['월평균기온', '월평균습도', '월평균_일교차', '월평균_PM25']

results = []
for code, sub_df in df_filtered.groupby('질병코드'):
    name = sub_df['질병명'].iloc[0] if '질병명' in sub_df.columns else code
    
    # 결측치 제거
    sub_df = sub_df.dropna(subset=['환자수', '월평균기온', '월평균습도', '월평균_일교차', '월평균_PM25', 'sin1', 'cos1'])
    
    if len(sub_df) < 12:
        print(f"⚠️ [스킵] 질병코드 {code}: 유효 데이터가 부족합니다 ({len(sub_df)}개).")
        continue
        
    # 5. 회귀 분석 수식 정의 (푸리에 항 적용)
    formula = '환자수 ~ 월평균기온 + 월평균습도 + 월평균_일교차 + 월평균_PM25 + sin1 + cos1'
    formula_null = '환자수 ~ sin1 + cos1'
    
    try:
        # [수정된 부분] GLM과 NegativeBinomial(alpha=1.0) 조합으로 .deviance 속성 보장 및 경고 해결
        model_full = smf.glm(formula, data=sub_df, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
        model_null = smf.glm(formula_null, data=sub_df, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
        
        # Deviance 기반 Marginal R^2 (D^2) 계산
        dev_full = model_full.deviance
        dev_null = model_null.deviance
        marginal_r2 = max(0, 1 - (dev_full / dev_null)) if dev_null > 0 else 0.0
        
        params = model_full.params
        
        # 6. 기상 변수 가중치 산출 (절댓값 기준 비율)
        w_temp = abs(params.get('월평균기온', 0))
        w_pm25 = abs(params.get('월평균_PM25', 0))
        w_humid = abs(params.get('월평균습도', 0))
        w_dtr = abs(params.get('월평균_일교차', 0))
        
        tot_w = w_temp + w_pm25 + w_humid + w_dtr
        if tot_w > 0:
            w_temp_pct = round((w_temp / tot_w) * 100, 2)
            w_pm25_pct = round((w_pm25 / tot_w) * 100, 2)
            w_humid_pct = round((w_humid / tot_w) * 100, 2)
            w_dtr_pct = round((w_dtr / tot_w) * 100, 2)
        else:
            w_temp_pct = w_pm25_pct = w_humid_pct = w_dtr_pct = 0.0

        results.append({
            '질병코드': code,
            '질병명': name,
            'Marginal_R2': round(marginal_r2, 4),
            '기온_가중치(%)': w_temp_pct,
            'PM25_가중치(%)': w_pm25_pct,
            '습도_가중치(%)': w_humid_pct,
            '일교차_가중치(%)': w_dtr_pct
        })
        
        # 7. Step 2 스코어링 연동을 위한 모델 및 메타데이터 저장
        model_save_data = {
            'params': params,
            'monthly_baseline': sub_df.groupby('월')[scale_target_cols].mean().to_dict(),
            'scaler_mean': sub_df[scale_target_cols].mean().to_dict(),
            'scaler_scale': sub_df[scale_target_cols].std().to_dict()
        }
        
        model_file_path = os.path.join(models_dir, f'model_{code}.pkl')
        with open(model_file_path, 'wb') as f:
            pickle.dump(model_save_data, f)
            
    except Exception as e:
        print(f"[에러] 질병코드 {code} ({name}) 분석 중 오류 발생: {e}")

# 8. 최종 결과 CSV 저장
if results:
    result_df = pd.DataFrame(results)
    output_csv = 'weather_weights_summary_2015_onwards.csv'
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n[성공] 모든 분석이 완료되었습니다. 결과 파일명: '{output_csv}'")
    print(result_df)
else:
    print("\n[실패] 저장된 분석 결과가 없습니다.")