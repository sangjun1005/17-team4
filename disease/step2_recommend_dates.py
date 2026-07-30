import os
import pickle
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

# 10일 예보 데이터 (8월 예시: 8/3, 8/4일에 미세먼지 및 폭염 집중)
dates = pd.date_range(start='2026-08-01', periods=10, freq='D')
forecast_data = {
    '날짜': dates.strftime('%Y-%m-%d'),
    '연도': [2026] * 10,
    '월': dates.month.tolist(),
    '월평균기온': [26.0, 26.5, 35.8, 34.5, 27.0, 25.8, 26.2, 26.8, 27.2, 26.0], # 8/3~8/4 폭염
    '월평균기압': [1012, 1011, 995, 998, 1010, 1012, 1011, 1010, 1012, 1011],  # 8/3~8/4 저기압
    '월최대_PM25': [18, 22, 78, 65, 25, 20, 18, 24, 28, 20],                  # 8/3~8/4 미세먼지 급증
    '월평균습도': [65, 62, 90, 85, 62, 60, 64, 66, 68, 62]
}
df_forecast = pd.DataFrame(forecast_data)

model_files = [f for f in os.listdir(models_dir) if f.startswith('gam_') and f.endswith('.pkl')]

disease_scores = {}
disease_names = {}

for m_file in model_files:
    code = m_file.replace('gam_', '').replace('.pkl', '')
    
    with open(os.path.join(models_dir, m_file), 'rb') as f:
        saved_info = pickle.load(f)
        
    gam = saved_info['model']
    scaler = saved_info['scaler']
    X_cols = saved_info['X_cols']
    weather_cols = saved_info['weather_cols']
    monthly_baseline = saved_info['monthly_baseline']
    last_year = saved_info['last_train_year']
    
    calibrated_scores = []
    
    for idx, row in df_forecast.iterrows():
        target_month = int(row['월'])
        
        # 1. 당일 예보 기상 데이터 (연도 변수는 학습 최신 연도로 고정하여 폭주 방지)
        X_daily = [last_year] + [row[col] for col in weather_cols]
        X_daily_scaled = scaler.transform([X_daily])
        pred_daily = gam.predict(X_daily_scaled)[0]
        
        # 2. 해당 월 평년 평균 기상 데이터 (Baseline)
        base_weather = monthly_baseline.get(target_month, {})
        X_base = [last_year] + [base_weather.get(col, row[col]) for col in weather_cols]
        X_base_scaled = scaler.transform([X_base])
        pred_base = gam.predict(X_base_scaled)[0]
        
        # 3. [핵심] 평년 기상 대비 당일 기상의 환자 변동 비율 (%)
        if pred_base > 0:
            change_ratio = (pred_daily - pred_base) / pred_base
        else:
            change_ratio = 0.0
            
        # 4. 0~100 스코어링 변환 (50점 = 평년 수준)
        score = 50.0 + (change_ratio * 100.0)
        score = np.clip(score, 10.0, 95.0) # 현실적인 범위 제한
        calibrated_scores.append(score)
        
    disease_scores[code] = np.round(calibrated_scores, 1)
    disease_names[code] = DISEASE_ACTION_MAP.get(code, {}).get('명칭', code)

# 결과 리포트 생성
df_report = pd.DataFrame({'날짜': df_forecast['날짜']})
for code, scores in disease_scores.items():
    df_report[f'{code}_점수'] = scores

score_cols = [c for c in df_report.columns if c.endswith('_점수')]
df_report['종합_추천점수'] = np.round(df_report[score_cols].mean(axis=1), 1)

# 위험 질환 산출 (기준 재조정: 70점 이상 = 🟠주의 / 80점 이상 = 🔴고위험)
high_risk_list = []
action_items_list = []

for idx, row in df_report.iterrows():
    danger_diseases = []
    actions = set()
    
    for code in disease_scores.keys():
        score = row[f'{code}_점수']
        d_name = disease_names[code]
        
        if score >= 80:
            danger_diseases.append(f"🔴{d_name}({score}점)")
            if code in DISEASE_ACTION_MAP:
                actions.add(DISEASE_ACTION_MAP[code]['지참물품'])
        elif score >= 70:
            danger_diseases.append(f"🟠{d_name}({score}점)")
            if code in DISEASE_ACTION_MAP:
                actions.add(DISEASE_ACTION_MAP[code]['지참물품'])
                
    if danger_diseases:
        high_risk_list.append(", ".join(danger_diseases))
        action_items_list.append(" / ".join(actions))
    else:
        high_risk_list.append("🟢 특이 위험 질환 없음 (평년 기상 수준)")
        action_items_list.append("기본 건강검진 장비 지참")

df_report['위험_질병군_목록'] = high_risk_list
df_report['순회진료_집중_준비물'] = action_items_list

print("==============================================================================================")
print("🗓️ [보정 완료] 기상 기여 위험도 지수 기반 10일 순회진료 추천 리포트")
print("==============================================================================================")
print(df_report[['날짜', '종합_추천점수', '위험_질병군_목록']].to_string(index=False))
print("==============================================================================================\n")

top3 = df_report.sort_values(by='종합_추천점수', ascending=False).head(3)
print("💡 🏆 [최종 추천] 순회진료 방문 우선순위 Top 3:")
for rank, (_, row) in enumerate(top3.iterrows(), 1):
    print(f"  {rank}위 : {row['날짜']} (종합점수: {row['종합_추천점수']}점)")
    print(f"        - 🚨 주의 질환 : {row['위험_질병군_목록']}")
    print(f"        - 🎒 권장 준비물 : {row['순회진료_집중_준비물']}\n")