import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터 로드
file_path = 'disease_weather_air_merged.csv'
df = pd.read_csv(file_path)

print("📌 [1] 데이터 로드 완료")

# 2. 보완 및 트렌드 보정 전처리 함수
def preprocess_disease_with_trends(data, filter_covid=True):
    temp_df = data.copy()
    
    # 날짜 처리
    temp_df['진료년월_dt'] = pd.to_datetime(temp_df['진료년월'].astype(str), format='%Y%m', errors='coerce')
    if temp_df['진료년월_dt'].isna().sum() > 0:
        temp_df['진료년월_dt'] = pd.to_datetime(temp_df['진료년월'].astype(str), errors='coerce')
        
    temp_df = temp_df.sort_values(by=['질병코드', '진료년월_dt']).reset_index(drop=True)
    
    # [Point 1] 시계열 트렌드 변수 생성 (연도, 월)
    temp_df['연도'] = temp_df['진료년월_dt'].dt.year
    temp_df['월'] = temp_df['진료년월_dt'].dt.month
    
    # [Point 2] 질병별 환자 수 기저치(Base) 보정 파생변수 생성
    # 전월 환자수(lag1) 및 전년 동월 환자수(lag12)
    temp_df['환자수_lag1'] = temp_df.groupby('질병코드')['환자수'].shift(1)
    temp_df['환자수_lag12'] = temp_df.groupby('질병코드')['환자수'].shift(12)
    
    # [Point 3] 기상 요인 Lag1 생성
    weather_cols = [
        '월평균기온', '월평균최저기온', '월평균최고기온', '월평균일교차', '월최대일교차', 
        '한파일수', '폭염일수', '월평균습도', '월총강수량', '월강수일수', '월평균기압', 
        '월평균_PM10', '월최대_PM10', 'PM10_나쁨일수', '월평균_PM25', '월최대_PM25'
    ]
    target_weather = [c for c in weather_cols if c in temp_df.columns]
    for col in target_weather:
        temp_df[f'{col}_lag1'] = temp_df[col].shift(1)
        
    # 코로나19 기간 (2020~2022) 필터링
    if filter_covid:
        covid_mask = (temp_df['진료년월_dt'] >= '2020-01-01') & (temp_df['진료년월_dt'] <= '2022-12-31')
        temp_df = temp_df[~covid_mask].reset_index(drop=True)
        
    # 결측치 제거
    temp_df = temp_df.dropna().reset_index(drop=True)
    print(f"📌 [2] 트렌드 보정 및 Lag 변수 생성 완료 (남은 데이터: {len(temp_df)}행)")
    print("="*60)
    
    return temp_df

df_clean = preprocess_disease_with_trends(df, filter_covid=True)

# 3. 고도화 분석 함수
def run_trend_adjusted_model(data, disease_code='M17'):
    sub_df = data[data['질병코드'] == disease_code].copy().reset_index(drop=True)
    disease_name = sub_df['질병명'].iloc[0] if '질병명' in sub_df.columns and not sub_df.empty else disease_code
    
    print(f"🔎 Target 질병 분석: [{disease_code}] {disease_name} (총 {len(sub_df)}개 월별 데이터)")
    
    if len(sub_df) < 15:
        print("⚠️ 학습 데이터 수가 부족합니다.")
        return None, None

    # 독립변수(X) 설정
    exclude_cols = [
        '질병코드', '질병명', '진료년월', '진료년월_dt', '환자수', 
        '내원일수', '청구건수', '요양급여비용총액', '보험자부담금'
    ]
    feature_cols = [c for c in sub_df.columns if c not in exclude_cols]
    
    X = sub_df[feature_cols]
    y = sub_df['환자수']
    y_log = np.log1p(y)
    
    # Train / Test Split (최근 15% 데이터를 테스트셋으로 사용)
    split_idx = int(len(sub_df) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_log, y_test_log = y_log.iloc[:split_idx], y_log.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    models = {
        'Ridge 회귀': Ridge(alpha=10.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.03, random_state=42)
    }
    
    results = {}
    print("\n📊 ===== [보정 후 모델 평가 결과 (실제 환자 수 기준)] =====")
    for name, model in models.items():
        model.fit(X_train, y_train_log)
        preds_log = model.predict(X_test)
        
        preds = np.expm1(preds_log)
        
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        
        results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
        print(f"[{name:15s}] R² 설명력: {r2:7.4f} | RMSE: {rmse:10.1f}명 | MAE: {mae:10.1f}명")
        
    # 기상 변수만의 상대적 중요도 추출 (상위 5개)
    best_xgb = models['XGBoost']
    importances = pd.Series(best_xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
    
    print("\n💡 ===== [영향력이 가장 큰 상위 5개 변수 (XGBoost)] =====")
    for idx, (col_name, imp) in enumerate(importances.head(5).items(), 1):
        print(f" {idx}. {col_name:22s}: {imp*100:.2f}%")
        
    return results, importances

# 실행
results, importances = run_trend_adjusted_model(df_clean, disease_code='M17')