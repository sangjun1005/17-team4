import os
import pickle
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
models_dir = os.path.join(BASE_DIR, 'saved_models')

DISEASE_ACTION_MAP = {
    'M17': {'명칭': '무릎관절증', '지참물품': '온열 파스/찜질팩, 관절 통증완화제'},
    'J45': {'명칭': '천식', '지참물품': '호흡기 흡입제, 방진마스크, 폐기능 측정기'},
    'J20': {'명칭': '급성기관지염', '지참물품': '진해거담제, 체온계, 진료 청진기'},
    'I10': {'명칭': '고혈압', '지참물품': '자동 혈압계, 복용 지도서'},
    'J30': {'명칭': '알레르기 비염', '지참물품': '항히스타민제, 비강 분무액'},
    'E11': {'명칭': '당뇨병', '지참물품': '혈당측정기, 당뇨 소모품'},
    'J06': {'명칭': '급성상기도감염', '지참물품': '해열진통제, 감기약'},
    'K29': {'명칭': '위염및위십이지장염', '지참물품': '소화제, 제산제'},
    'L23': {'명칭': '접촉성피부염', '지참물품': '피부연고, 항히스타민제'},
    'M54': {'명칭': '등통증', '지참물품': '근육이완제, 소염진통제'}
}

def predict_from_params(params, year_scaled, month, temp, pm25, humid, used_lag=False):
    linear_pred = params.get('Intercept', 0.0)
    linear_pred += params.get('연도_scaled', 0.0) * year_scaled
    
    if month > 1:
        month_key = f'C(월)[T.{month}]'
        linear_pred += params.get(month_key, 0.0)
        
    if used_lag and '환자수_lag12' in params:
        # 직전 연도 동월 평균치 추정 대입
        linear_pred += params.get('환자수_lag12', 0.0) * 100000 
        
    t_key = '월평균기온_lag1' if '월평균기온_lag1' in params else '월평균기온'
    p_key = '월최대_PM25_lag1' if '월최대_PM25_lag1' in params else '월최대_PM25'
    h_key = '월평균습도_lag1' if '월평균습도_lag1' in params else '월평균습도'
    
    linear_pred += params.get(t_key, 0.0) * temp
    linear_pred += params.get(p_key, 0.0) * pm25
    linear_pred += params.get(h_key, 0.0) * humid
    
    return np.exp(linear_pred)

# 10일 예보 데이터 (여름철 폭염/미세먼지 변동 시뮬레이션 포함)
dates = pd.date_range(start='2026-08-01', periods=10, freq='D')
forecast_data = {
    '날짜': dates.strftime('%Y-%m-%d'),
    '연도': [2026] * 10,
    '월': dates.month.tolist(),
    '월평균기온': [26.0, 26.5, 35.5, 34.2, 27.0, 25.8, 26.2, 26.8, 27.2, 26.0],
    '월최대_PM25': [18, 22, 95, 82, 25, 20, 18, 24, 28, 20],
    '월평균습도': [65, 62, 90, 85, 62, 60, 64, 66, 68, 62]
}
df_forecast = pd.DataFrame(forecast_data)

model_files = [f for f in os.listdir(models_dir) if f.startswith('model_') and f.endswith('.pkl')]

disease_scores = {}
disease_names = {}

for m_file in model_files:
    code = m_file.replace('model_', '').replace('.pkl', '')
    
    with open(os.path.join(models_dir, m_file), 'rb') as f:
        saved_info = pickle.load(f)
        
    params = saved_info['params']
    monthly_baseline = saved_info['monthly_baseline']
    last_year = saved_info['last_train_year']
    min_year = saved_info['min_train_year']
    used_lag = saved_info.get('used_lag', False)
    
    year_scaled = last_year - min_year
    calibrated_scores = []
    
    for idx, row in df_forecast.iterrows():
        target_month = int(row['월'])
        
        pred_daily = predict_from_params(
            params, year_scaled, target_month,
            row['월평균기온'], row['월최대_PM25'], row['월평균습도'], used_lag
        )
        
        base_w = monthly_baseline.get(target_month, {})
        pred_base = predict_from_params(
            params, year_scaled, target_month,
            base_w.get('월평균기온', row['월평균기온']),
            base_w.get('월최대_PM25', row['월최대_PM25']),
            base_w.get('월평균습도', row['월평균습도']), used_lag
        )
        
        change_ratio = (pred_daily - pred_base) / pred_base if pred_base > 0 else 0.0
        # 점수 민감도 상향 조정 (변동폭이 뚜렷하게 나타나도록 튜닝)
        score = np.clip(50.0 + (change_ratio * 150.0), 10.0, 95.0)
        calibrated_scores.append(score)
        
    disease_scores[code] = np.round(calibrated_scores, 1)
    disease_names[code] = DISEASE_ACTION_MAP.get(code, {}).get('명칭', code)

df_report = pd.DataFrame({'날짜': df_forecast['날짜']})
for code, scores in disease_scores.items():
    df_report[f'{code}_점수'] = scores

score_cols = [c for c in df_report.columns if c.endswith('_점수')]
df_report['종합_추천점수'] = np.round(df_report[score_cols].mean(axis=1), 1)

high_risk_list = []
action_items_list = []

for idx, row in df_report.iterrows():
    danger_diseases = []
    actions = set()
    
    for code in disease_scores.keys():
        score = row[f'{code}_점수']
        d_name = disease_names.get(code, code)
        if score >= 80:
            danger_diseases.append(f"[고위험]{d_name}({score}점)")
            if code in DISEASE_ACTION_MAP:
                actions.add(DISEASE_ACTION_MAP[code]['지참물품'])
        elif score >= 65:
            danger_diseases.append(f"[주의]{d_name}({score}점)")
            if code in DISEASE_ACTION_MAP:
                actions.add(DISEASE_ACTION_MAP[code]['지참물품'])
                
    if danger_diseases:
        high_risk_list.append(", ".join(danger_diseases))
        action_items_list.append(" / ".join(actions))
    else:
        high_risk_list.append("특이 위험 질환 없음 (평년 기상 수준)")
        action_items_list.append("기본 건강검진 장비 지참")

df_report['위험_질병군_목록'] = high_risk_list
df_report['순회진료_집중_준비물'] = action_items_list

print("==============================================================================================")
print("10일 예보 기반 순회진료 추천 리포트 (시차 및 변동폭 보정본)")
print("==============================================================================================")
print(df_report[['날짜', '종합_추천점수', '위험_질병군_목록']].to_string(index=False))
print("==============================================================================================\n")