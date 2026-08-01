import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터 로드
file_path = 'disease_weather_air_merged.csv'
df = pd.read_csv(file_path)

print("📌 [1] 데이터 로드 완료")

# 2. 전처리 및 트렌드/기상 변수 분리 파이프라인
def preprocess_for_hybrid(data, filter_covid=True):
    temp_df = data.copy()
    
    # 날짜 처리
    temp_df['진료년월_dt'] = pd.to_datetime(temp_df['진료년월'].astype(str), format='%Y%m', errors='coerce')
    if temp_df['진료년월_dt'].isna().sum() > 0:
        temp_df['진료년월_dt'] = pd.to_datetime(temp_df['진료년월'].astype(str), errors='coerce')
        
    temp_df = temp_df.sort_values(by=['질병코드', '진료년월_dt']).reset_index(drop=True)
    
    # [트렌드 변수]
    temp_df['연도'] = temp_df['진료년월_dt'].dt.year
    temp_df['월'] = temp_df['진료년월_dt'].dt.month
    temp_df['환자수_lag1'] = temp_df.groupby('질병코드')['환자수'].shift(1)
    temp_df['환자수_lag12'] = temp_df.groupby('질병코드')['환자수'].shift(12)
    
    # [기상/대기 변수 + Lag1]
    weather_cols = [
        '월평균기온', '월평균최저기온', '월평균최고기온', '월평균일교차', '월최대일교차', 
        '한파일수', '폭염일수', '월평균습도', '월총강수량', '월강수일수', '월평균기압', 
        '월평균_PM10', '월최대_PM10', 'PM10_나쁨일수', '월평균_PM25', '월최대_PM25'
    ]
    target_weather = [c for c in weather_cols if c in temp_df.columns]
    for col in target_weather:
        temp_df[f'{col}_lag1'] = temp_df[col].shift(1)
        
    # 코로나19 기간 제거
    if filter_covid:
        covid_mask = (temp_df['진료년월_dt'] >= '2020-01-01') & (temp_df['진료년월_dt'] <= '2022-12-31')
        temp_df = temp_df[~covid_mask].reset_index(drop=True)
        
    temp_df = temp_df.dropna().reset_index(drop=True)
    print(f"📌 [2] 하이브리드 전처리 완료 (남은 데이터: {len(temp_df)}행)")
    print("="*65)
    return temp_df

df_clean = preprocess_for_hybrid(df, filter_covid=True)

# 3. 하이브리드 분석 모델 구동 함수
def run_hybrid_model(data, disease_code='M17'):
    sub_df = data[data['질병코드'] == disease_code].copy().reset_index(drop=True)
    disease_name = sub_df['질병명'].iloc[0] if '질병명' in sub_df.columns and not sub_df.empty else disease_code
    
    print(f"🔎 Target 질병 분석: [{disease_code}] {disease_name} (총 {len(sub_df)}개 월별 데이터)")
    
    # 변수 세트 정의
    trend_features = ['연도', '월', '환자수_lag1', '환자수_lag12']
    
    exclude_cols = ['질병코드', '질병명', '진료년월', '진료년월_dt', '환자수', 
                    '내원일수', '청구건수', '요양급여비용총액', '보험자부담금'] + trend_features
    weather_features = [c for c in sub_df.columns if c not in exclude_cols]
    
    X_trend = sub_df[trend_features]
    X_weather = sub_df[weather_features]
    y = sub_df['환자수']
    
    # Train / Test Split (시계열 유지)
    split_idx = int(len(sub_df) * 0.85)
    
    X_tr_trend, X_te_trend = X_trend.iloc[:split_idx], X_trend.iloc[split_idx:]
    X_tr_weath, X_te_weath = X_weather.iloc[:split_idx], X_weather.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # -------------------------------------------------------------
    # [1단계] 거시적 트렌드 모델 (Ridge)
    # -------------------------------------------------------------
    stage1_model = Ridge(alpha=10.0)
    stage1_model.fit(X_tr_trend, np.log1p(y_train))
    
    y_tr_pred_trend = np.expm1(stage1_model.predict(X_tr_trend))
    y_te_pred_trend = np.expm1(stage1_model.predict(X_te_trend))
    
    # Train 및 Test 잔차(Residual) 계산
    residuals_train = y_train - y_tr_pred_trend
    
    # -------------------------------------------------------------
    # [2단계] 미시적 기상 잔차 모델 (XGBoost)
    # -------------------------------------------------------------
    stage2_model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.03, random_state=42)
    stage2_model.fit(X_tr_weath, residuals_train)
    
    residuals_te_pred = stage2_model.predict(X_te_weath)
    
    # -------------------------------------------------------------
    # [3단계] 최종 결합 예측 (Stage 1 + Stage 2)
    # -------------------------------------------------------------
    y_final_pred = y_te_pred_trend + residuals_te_pred
    
    # 단일 XGBoost 비교용 (전체 변수로 바로 학습한 경우)
    single_xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.03, random_state=42)
    single_xgb.fit(pd.concat([X_tr_trend, X_tr_weath], axis=1), np.log1p(y_train))
    single_xgb_pred = np.expm1(single_xgb.predict(pd.concat([X_te_trend, X_te_weath], axis=1)))

    # 성능 평가
    r2_stage1 = r2_score(y_test, y_te_pred_trend)
    r2_single_xgb = r2_score(y_test, single_xgb_pred)
    r2_hybrid = r2_score(y_test, y_final_pred)
    
    rmse_hybrid = np.sqrt(mean_squared_error(y_test, y_final_pred))
    mae_hybrid = mean_absolute_error(y_test, y_final_pred)
    
    print("\n📊 ===== [단일 모델 vs 2-Stage 하이브리드 모델 성능 비교] =====")
    print(f" 1. 단일 XGBoost 모델 R²        : {r2_single_xgb:7.4f}")
    print(f" 2. 단일 Ridge(트렌드) 모델 R²   : {r2_stage1:7.4f}")
    print(f" 3. ★ [하이브리드 결합 모델] R²  : {r2_hybrid:7.4f} (RMSE: {rmse_hybrid:.1f}명, MAE: {mae_hybrid:.1f}명)")
    
    # 2단계 기상 잔차 예측에서의 핵심 기상 요인 중요도
    importances = pd.Series(stage2_model.feature_importances_, index=weather_features).sort_values(ascending=False)
    
    print("\n💡 ===== [기상/대기 요인 순수 영향력 상위 5개 (Stage 2 XGBoost)] =====")
    for idx, (col_name, imp) in enumerate(importances.head(5).items(), 1):
        print(f" {idx}. {col_name:22s}: {imp*100:.2f}%")
        
    return r2_hybrid, importances

# 분석 실행 (예: 무릎관절증 M17)
hybrid_r2, weather_imp = run_hybrid_model(df_clean, disease_code='J45')