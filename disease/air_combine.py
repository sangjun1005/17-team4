import os
import glob
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

AIR_FOLDER = './air_data'
DISEASE_WEATHER_FILE = './disease_weather_merged.csv'

air_files = glob.glob(os.path.join(AIR_FOLDER, "*.csv")) + \
            glob.glob(os.path.join(AIR_FOLDER, "*.xlsx")) + \
            glob.glob(os.path.join(AIR_FOLDER, "*.xls"))

air_files = [f for f in air_files if not os.path.basename(f).startswith('~$')]

if not air_files:
    print("Error: 분석 대상 파일이 없습니다.")
    exit()

print(f"Total files found: {len(air_files)}")

def load_file(fpath):
    ext = os.path.splitext(fpath)[1].lower()
    
    # 1. Excel 파일 로드 시도
    if ext in ['.xlsx', '.xls']:
        for engine in ['openpyxl', 'xlrd', None]:
            try:
                return pd.read_excel(fpath, engine=engine)
            except Exception:
                continue
        
        # 텍스트 형태 파일 가능성 시도
        for enc in ['utf-8-sig', 'cp949', 'euc-kr', 'utf-8']:
            try:
                return pd.read_csv(fpath, encoding=enc, sep=None, engine='python')
            except Exception:
                continue
                
    # 2. CSV 파일 로드 시도
    else:
        for enc in ['utf-8-sig', 'cp949', 'euc-kr', 'utf-8']:
            try:
                return pd.read_csv(fpath, encoding=enc, sep=None, engine='python')
            except Exception:
                continue
                
    return None

daily_records = []

for idx, fpath in enumerate(air_files, 1):
    fname = os.path.basename(fpath)
    df = load_file(fpath)

    if df is None or df.empty:
        print(f"[{idx}/{len(air_files)}] Load failed or empty: {fname}")
        continue

    # 컬럼명 정리
    df.columns = [str(c).strip().replace(' ', '') for c in df.columns]

    # 천안시/충남 지역 데이터 필터링
    region_col = next((c for c in df.columns if any(k in c for k in ['지역', '주소', '측정소명'])), None)
    if region_col:
        df = df[df[region_col].astype(str).str.contains('천안|충남', na=False)]
    
    if df.empty:
        continue

    # 날짜 컬럼 추출
    date_col = next((c for c in df.columns if any(k in c for k in ['측정일시', '일시', '날짜', 'TM'])), None)
    if not date_col:
        continue

    # YYYYMMDD 규격화
    df['date_str'] = df[date_col].astype(str).str.replace(r'[^0-9]', '', regex=True).str[:8]
    df['date_dt'] = pd.to_datetime(df['date_str'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date_dt'])

    if df.empty:
        continue

    # 미세먼지 컬럼 추출
    pm10_col = next((c for c in df.columns if 'PM10' in c or '미세먼지' in c), None)
    pm25_col = next((c for c in df.columns if 'PM25' in c or 'PM2.5' in c or '초미세' in c), None)

    # 수치형 변환 및 이상치(-999, 9999 등) 처리
    if pm10_col:
        df['PM10_num'] = pd.to_numeric(df[pm10_col].astype(str).str.replace(',', ''), errors='coerce')
        df.loc[(df['PM10_num'] < 0) | (df['PM10_num'] > 1000), 'PM10_num'] = np.nan
    else:
        df['PM10_num'] = np.nan

    if pm25_col:
        df['PM25_num'] = pd.to_numeric(df[pm25_col].astype(str).str.replace(',', ''), errors='coerce')
        df.loc[(df['PM25_num'] < 0) | (df['PM25_num'] > 1000), 'PM25_num'] = np.nan
    else:
        df['PM25_num'] = np.nan

    # 유효 데이터 수집
    valid_df = df[['date_dt', 'PM10_num', 'PM25_num']].dropna(how='all', subset=['PM10_num', 'PM25_num'])
    if not valid_df.empty:
        daily_records.append(valid_df)
        print(f"[{idx}/{len(air_files)}] Success: {fname}")

if not daily_records:
    print("Error: 유효한 대기질 데이터가 추출되지 않았습니다.")
    exit()

# 일 단위 및 월 단위 집계
raw_air_df = pd.concat(daily_records, ignore_index=True)

daily_summary = raw_air_df.groupby('date_dt').agg(
    PM10=('PM10_num', 'mean'),
    PM25=('PM25_num', 'mean')
).reset_index()

daily_summary['진료년월'] = daily_summary['date_dt'].dt.strftime('%Y-%m')
daily_summary['PM10_bad_day'] = (daily_summary['PM10'] >= 81).astype(int)

monthly_air = daily_summary.groupby('진료년월').agg(
    월평균_PM10=('PM10', 'mean'),
    월최대_PM10=('PM10', 'max'),
    PM10_나쁨일수=('PM10_bad_day', 'sum'),
    월평균_PM25=('PM25', 'mean'),
    월최대_PM25=('PM25', 'max')
).reset_index().round(2)

print(f"Summary completed: {len(monthly_air)} months ({monthly_air['진료년월'].min()} ~ {monthly_air['진료년월'].max()})")

# 병합 및 저장
if os.path.exists(DISEASE_WEATHER_FILE):
    dw_df = pd.read_csv(DISEASE_WEATHER_FILE)
    final_df = pd.merge(dw_df, monthly_air, on='진료년월', how='left')
    
    if '질병명' in final_df.columns:
        final_df = final_df.sort_values(by=['질병명', '진료년월']).reset_index(drop=True)

    final_csv = './disease_weather_air_merged.csv'
    final_excel = './disease_weather_air_merged.xlsx'

    final_df.to_csv(final_csv, index=False, encoding='utf-8-sig')
    final_df.to_excel(final_excel, index=False)

    print(f"Saved: {final_csv} ({len(final_df)} rows)")
else:
    monthly_air.to_csv('./monthly_air_summary.csv', index=False, encoding='utf-8-sig')
    print("Saved: ./monthly_air_summary.csv")