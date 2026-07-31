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
file_path = os.path.join(BASE_DIR, 'disease_weather_air_merged.csv')
models_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(models_dir, exist_ok=True)

if not os.path.exists(file_path):
    alt_path = os.path.join(BASE_DIR, './combined_data/disease_weather_air_merged.csv')
    if os.path.exists(alt_path):
        file_path = alt_path
    else:
        print(f"파일 없음")
        exit()

df = pd.read_csv(file_path)

df['진료년월_dt'] = pd.to_datetime(df['진료년월'].astype(str), errors='coerce')
df['연도'] = df['진료년월_dt'].dt.year
df['월'] = df['진료년월_dt'].dt.month

df['sin1'] = np.sin(2 * np.pi * df['월'] / 12)
df['cos1'] = np.cos(2 * np.pi * df['월'] / 12)

df_filtered = df[(df['연도'] >= 2015) & (~df['연도'].between(2020, 2022))].copy()

scale_target_cols = ['월평균기온', '월평균습도', '월평균_일교차', '월평균_PM25']

results = []
for code, sub_df in df_filtered.groupby('질병코드'):
    name = sub_df['질병명'].iloc[0] if '질병명' in sub_df.columns else code
    
    sub_df = sub_df.dropna(subset=['환자수', '월평균기온', '월평균습도', '월평균_일교차', '월평균_PM25', 'sin1', 'cos1'])
    
    if len(sub_df) < 12:
        print(f"유효 데이터 부족")
        continue
        
    formula = '환자수 ~ 월평균기온 + 월평균습도 + 월평균_일교차 + 월평균_PM25 + sin1 + cos1'
    formula_null = '환자수 ~ sin1 + cos1'
    
    try:
        model_full = smf.glm(formula, data=sub_df, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
        model_null = smf.glm(formula_null, data=sub_df, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
        
        dev_full = model_full.deviance
        dev_null = model_null.deviance
        marginal_r2 = max(0, 1 - (dev_full / dev_null)) if dev_null > 0 else 0.0
        
        params = model_full.params
        
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
        print(f"[에러] {e}")

if results:
    result_df = pd.DataFrame(results)
    output_csv = 'weather_weights_summary_2015_onwards.csv'
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n[성공] '{output_csv}'")
    print(result_df)
else:
    print("\n[실패]")