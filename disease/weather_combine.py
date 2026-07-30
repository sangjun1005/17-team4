import os
import glob
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 0. 경로 설정
# ---------------------------------------------------------
WEATHER_FOLDER = './weather_data'
OUTPUT_FILE = './monthly_weather_summary.csv'

weather_files = glob.glob(os.path.join(WEATHER_FOLDER, "*.csv")) + \
                glob.glob(os.path.join(WEATHER_FOLDER, "*.xlsx")) + \
                glob.glob(os.path.join(WEATHER_FOLDER, "*.xls"))

weather_files = [f for f in weather_files if not os.path.basename(f).startswith('~$')]

if not weather_files:
    print("❌ [오류] weather_data 폴더 내 분석 대상 기상 파일이 없습니다.")
    exit()

print(f"📂 총 {len(weather_files)}개의 기상 데이터 파일을 발견했습니다. 전처리를 시작합니다...")

# ---------------------------------------------------------
# 1. 파일별 로드 및 컬럼 표준화
# ---------------------------------------------------------
raw_weather_list = []

for fpath in weather_files:
    try:
        if fpath.endswith('.csv'):
            try:
                temp_df = pd.read_csv(fpath, encoding='utf-8-sig')
            except UnicodeDecodeError:
                temp_df = pd.read_csv(fpath, encoding='cp949')
        else:
            temp_df = pd.read_excel(fpath)
        
        # 공백 제거 및 문자열 표준화
        temp_df.columns = [str(c).strip().replace(' ', '') for c in temp_df.columns]
        raw_weather_list.append(temp_df)
    except Exception as e:
        print(f"⚠️ 읽기 오류 발생 ({os.path.basename(fpath)}): {e}")

if not raw_weather_list:
    print("❌ 데이터 읽기 실패")
    exit()

weather_df = pd.concat(raw_weather_list, ignore_index=True)

# ---------------------------------------------------------
# 2. 필수 기상 변수 컬럼 매핑 (자동 검색)
# ---------------------------------------------------------
def find_column(df, candidates):
    for cand in candidates:
        for col in df.columns:
            if cand in col:
                return col
    return None

date_col  = find_column(weather_df, ['일시', '날짜', 'TM', 'STD_DAY', '일자'])
avg_temp_col = find_column(weather_df, ['평균기온', 'TA_AVG', 'AVG_TEMP'])
min_temp_col = find_column(weather_df, ['최저기온', 'TA_MIN', 'MIN_TEMP'])
max_temp_col = find_column(weather_df, ['최고기온', 'TA_MAX', 'MAX_TEMP'])
humid_col = find_column(weather_df, ['상대습도', '평균습도', 'HM_AVG', 'HUMIDITY', '습도'])
precip_col = find_column(weather_df, ['일강수량', '강수량', 'RN_DAY'])
press_col = find_column(weather_df, ['현지기압', '해면기압', '기압', 'PA_AVG'])

print("\n🔍 매핑된 기상 컬럼 목록:")
print(f" - 날짜: {date_col}")
print(f" - 평균기온: {avg_temp_col}")
print(f" - 최저기온: {min_temp_col}")
print(f" - 최고기온: {max_temp_col}")
print(f" - 습도: {humid_col}")
print(f" - 강수량: {precip_col}")

if not (date_col and avg_temp_col and min_temp_col and max_temp_col):
    print("❌ [오류] 기온(평균/최저/최고) 또는 날짜 컬럼을 찾을 수 없습니다.")
    exit()

# 수치형 데이터 변환
numeric_cols = [avg_temp_col, min_temp_col, max_temp_col]
if humid_col: numeric_cols.append(humid_col)
if precip_col: numeric_cols.append(precip_col)
if press_col: numeric_cols.append(press_col)

for col in numeric_cols:
    weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce')

# 강수량 결측치는 0mm 처리
if precip_col:
    weather_df[precip_col] = weather_df[precip_col].fillna(0.0)

# ---------------------------------------------------------
# 3. [핵심] 일교차(DTR) 및 극단 기상 변수 산출
# ---------------------------------------------------------
# 일교차 = 최고기온 - 최저기온
weather_df['일교차'] = weather_df[max_temp_col] - weather_df[min_temp_col]

# 음수 일교차(이상치) 보정
weather_df['일교차'] = weather_df['일교차'].apply(lambda x: x if x >= 0 else np.nan)

# 극단 기상 플래그
weather_df['한파발생'] = (weather_df[min_temp_col] <= -12.0).astype(int)
weather_df['폭염발생'] = (weather_df[max_temp_col] >= 33.0).astype(int)
if precip_col:
    weather_df['강수발생'] = (weather_df[precip_col] >= 1.0).astype(int)
else:
    weather_df['강수발생'] = 0

# 날짜 규격화 (YYYYMMDD)
weather_df['date_dt'] = pd.to_datetime(
    weather_df[date_col].astype(str).str.replace(r'[^0-9]', '', regex=True).str[:8],
    format='%Y%m%d', errors='coerce'
)
weather_df = weather_df.dropna(subset=['date_dt'])

# ---------------------------------------------------------
# 4. 1단계: 전국/지역 일별 평균 및 극단 기상 집계
# ---------------------------------------------------------
agg_dict = {
    '전국_평균기온': (avg_temp_col, 'mean'),
    '전국_최저기온': (min_temp_col, 'mean'),
    '전국_최고기온': (max_temp_col, 'mean'),
    '전국_일교차': ('일교차', 'mean'),
    '전국_최대일교차': ('일교차', 'max'),
    '한파관측비율': ('한파발생', 'mean'),
    '폭염관측비율': ('폭염발생', 'mean'),
    '강수관측비율': ('강수발생', 'mean')
}

if humid_col: agg_dict['전국_습도'] = (humid_col, 'mean')
if precip_col: agg_dict['전국_강수량'] = (precip_col, 'mean')
if press_col: agg_dict['전국_기압'] = (press_col, 'mean')

daily_nat_weather = weather_df.groupby('date_dt').agg(**agg_dict).reset_index()
daily_nat_weather['진료년월'] = daily_nat_weather['date_dt'].dt.strftime('%Y-%m')

# ---------------------------------------------------------
# 5. 2단계: 월별(진료년월 YYYY-MM) 최종 요약 집계
# ---------------------------------------------------------
monthly_agg_dict = {
    '월평균기온': ('전국_평균기온', 'mean'),
    '월평균_최저기온': ('전국_최저기온', 'mean'),
    '월평균_최고기온': ('전국_최고기온', 'mean'),
    '월평균_일교차': ('전국_일교차', 'mean'),
    '월최대_일교차': ('전국_최대일교차', 'max'),
    '한파일수_비율': ('한파관측비율', 'mean'),
    '폭염일수_비율': ('폭염관측비율', 'mean')
}

if humid_col: monthly_agg_dict['월평균습도'] = ('전국_습도', 'mean')
if precip_col: monthly_agg_dict['월총강수량'] = ('전국_강수량', 'sum')
if press_col: monthly_agg_dict['월평균기압'] = ('전국_기압', 'mean')

monthly_weather = daily_nat_weather.groupby('진료년월').agg(**monthly_agg_dict).reset_index()

# 소수점 반올림 정리
float_cols = monthly_weather.select_dtypes(include=[np.number]).columns
monthly_weather[float_cols] = monthly_weather[float_cols].round(2)

# 저장
monthly_weather.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print("\n" + "="*80)
print(f"✅ [완료] 월별 기상 종합 데이터 생성 성공: {OUTPUT_FILE}")
print("="*80)
print(monthly_weather.head(10))
print("="*80)