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

# 계수 기반 예측 계산 함수 (patsy 피클 로딩 문제 완전 해결)
def predict_from_params(params, year_scaled, month, temp, pm25, humid):
    linear_pred = params.get('Intercept', 0.0)
    linear_pred += params.get('연도_scaled', 0.0) * year_scaled
    
    # 월 범주형 변수(Dummy) 계수 적용
    if month > 1:
        month_key = f'C(월)[T.{month}]'
        linear_pred += params.get(month_key, 0.0)
        
    linear_pred += params.get('월평균기온', 0.0) * temp
    linear_pred += params.get('월최대_PM25', 0.0) * pm25
    linear_pred += params.get('월평균습도', 0.0) * humid
    
    return np.exp(linear_pred)

# 10일 예보 데이터
dates = pd.date_range(start='2026-08-01', periods=10, freq='D')
forecast_data = {
    '날짜': dates.strftime('%Y-%m-%d'),
    '연도': [2026] * 10,
    '월': dates.month.tolist(),
    '월평균기온': [26.0, 26.5, 33.8, 32.5, 27.0, 25.8, 26.2, 26.8, 27.2, 26.0],
    '월최대_PM25': [18, 22, 78, 65, 25, 20, 18, 24, 28, 20],
    '월평균습도': [65, 62, 88, 82, 62, 60, 64, 66, 68, 62]
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
    
    year_scaled = last_year - min_year
    calibrated_scores = []
    
    for idx, row in df_forecast.iterrows():
        target_month = int(row['월'])
        
        # 1. 당일 기상 기반 예측값
        pred_daily = predict_from_params(
            params, year_scaled, target_month,
            row['월평균기온'], row['월최대_PM25'], row['월평균습도']
        )
        
        # 2. 평년 동월 Baseline 기반 예측값
        base_w = monthly_baseline.get(target_month, {})
        pred_base = predict_from_params(
            params, year_scaled, target_month,
            base_w.get('월평균기온', row['월평균기온']),
            base_w.get('월최대_PM25', row['월최대_PM25']),
            base_w.get('월평균습도', row['월평균습도'])
        )
        
        # 3. 변동 비율 기반 위험도 점수 계산
        change_ratio = (pred_daily - pred_base) / pred_base if pred_base > 0 else 0.0
        score = np.clip(50.0 + (change_ratio * 100.0), 10.0, 95.0)
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
        if score >= 85:
            danger_diseases.append(f"[고위험]{d_name}({score}점)")
            if code in DISEASE_ACTION_MAP:
                actions.add(DISEASE_ACTION_MAP[code]['지참물품'])
        elif score >= 70:
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
print("10일 예보 기반 순회진료 추천 리포트")
print("==============================================================================================")
print(df_report[['날짜', '종합_추천점수', '위험_질병군_목록']].to_string(index=False))
print("==============================================================================================\n")