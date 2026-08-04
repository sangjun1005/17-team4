import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# [설정] API 키 및 기본 경로 설정
# ==========================================
# ※ 공공데이터포털에서 받은 인증키를 그대로 넣으세요. (이중 인코딩 방지 로직이 적용됩니다)
SERVICE_KEY = "YOUR_PUBLIC_DATA_API_KEY_HERE"  
WEIGHTS_CSV_PATH = "weather_weights_summary_2015_onwards.csv"  # 1단계 결과 파일

# ==========================================
# 공통: 이중 인코딩 및 타임아웃 방어형 요청 함수
# ==========================================
def safe_api_request(url, params, max_retries=3, timeout=15):
    """
    1. 이중 인코딩 방지 (serviceKey가 변조되지 않도록 처리)
    2. 브라우저 위장 User-Agent 헤더 추가
    3. Read timed out 발생 시 자동 재시도
    """
    # 브라우저와 동일한 헤더 설정 (방화벽 차단 방지)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    # serviceKey가 포함되어 있다면 이중 인코딩 방지를 위해 unquote 처리 후 requests에 맡김
    if 'serviceKey' in params and params['serviceKey']:
        params['serviceKey'] = requests.utils.unquote(params['serviceKey'])

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response
            else:
                print(f"[경고] API 응답 코드 비정상 ({response.status_code}), 재시도 중... ({attempt}/{max_retries})")
        except requests.exceptions.Timeout:
            print(f"[경고] API 응답 시간 초과(Read timed out). 재시도 중... ({attempt}/{max_retries})")
        except requests.exceptions.RequestException as e:
            print(f"[경고] 통신 에러 발생: {e}. 재시도 중... ({attempt}/{max_retries})")
        
        if attempt < max_retries:
            time.sleep(2)
            
    print(f"[오류] 최대 재시도 횟수({max_retries}회) 초과. API 호출 실패.")
    return None

# ==========================================
# 1. 에어코리아 실시간 측정소별 미세먼지 조회 API
# ==========================================
def fetch_realtime_pm25(service_key, station_name="성성동"):
    url = "http://apis.data.go.kr/B552584/ArpltnInqireSvc/getMsrstnAcctoRltmMesureDnsty"
    params = {
        'serviceKey': service_key,
        'returnType': 'json',
        'numOfRows': 24,
        'pageNo': 1,
        'stationName': station_name,
        'dataTerm': 'DAILY',
        'ver': '1.3'
    }
    
    print(f"[정보] 에어코리아 실시간 측정소({station_name}) 데이터 조회 요청 중...")
    response = safe_api_request(url, params)
    if not response:
        print(f"[안내] 실시간 API 통신 실패로 오늘 기본값(20.0)을 적용합니다.")
        return 20.0
        
    try:
        res_json = response.json()
        items = res_json.get('response', {}).get('body', {}).get('items', [])
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        for item in items:
            data_time = item.get('dataTime', '')
            pm25_value = item.get('pm25Value')
            if today_str in data_time and pm25_value is not None:
                if str(pm25_value).strip() not in ["-", "", "None"]:
                    print(f"[성공] 실시간 관측 미세먼지 수집 완료 (시간: {data_time}, PM2.5: {pm25_value} ㎍/㎥)")
                    return float(pm25_value)
                    
        for item in items:
            pm25_value = item.get('pm25Value')
            if pm25_value is not None and str(pm25_value).strip() not in ["-", "", "None"]:
                print(f"[안내] 가장 최근 유효 측정값 사용 (시간: {item.get('dataTime')}, PM2.5: {pm25_value})")
                return float(pm25_value)
                
    except Exception as e:
        print(f"[에러] 실시간 데이터 파싱 오류: {e}")
        
    return 20.0

# ==========================================
# 2. 에어코리아 미세먼지 예보 API (미래 2~7일차용)
# ==========================================
def fetch_pm25_forecast(service_key):
    url = "http://apis.data.go.kr/B552584/ArpltnInqireSvc/getMinuDustFrcstDspth"
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    params = {
        'serviceKey': service_key,
        'returnType': 'json',
        'numOfRows': 10,
        'pageNo': 1,
        'searchDate': today_str,
        'informCode': 'PM25'
    }
    
    print(f"[정보] 에어코리아 미세먼지 예보 데이터 조회 요청 중...")
    response = safe_api_request(url, params)
    items = []
    
    if response:
        res_json = response.json()
        items = res_json.get('response', {}).get('body', {}).get('items', [])
        
    if not items:
        params.pop('searchDate', None)
        response = safe_api_request(url, params)
        if response:
            res_json = response.json()
            items = res_json.get('response', {}).get('body', {}).get('items', [])
            
    if items:
        print(f"[성공] 미세먼지 예보 데이터 수집 완료")
        return items[0]
        
    return None

def parse_pm25_to_dict(forecast_item, target_region="충남"):
    grade_mapping = {'낮음': 20.0, '높음': 50.0, '보통': 35.0, '나쁨': 75.0, '매우나쁨': 120.0}
    pm_dict = {}
    if not forecast_item:
        return pm_dict
        
    date_keys = [('frcstOneDt', 'frcstOneCn'), ('frcstTwoDt', 'frcstTwoCn'), ('frcstThreeDt', 'frcstThreeCn'), ('frcstFourDt', 'frcstFourCn')]
    for dt_key, cn_key in date_keys:
        date_str = forecast_item.get(dt_key)
        content_str = forecast_item.get(cn_key, '')
        if date_str:
            target_grade = '낮음'
            for r_info in content_str.split(','):
                if target_region in r_info:
                    parts = r_info.split(':')
                    if len(parts) == 2:
                        target_grade = parts[1].strip()
                    break
            val = 20.0
            for term, score in grade_mapping.items():
                if term in target_grade:
                    val = score
                    break
            pm_dict[date_str] = val
    return pm_dict

# ==========================================
# 3. 기상청 예보 데이터 시뮬레이션 (7일)
# ==========================================
def get_weather_forecast_7days():
    dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    np.random.seed(42)
    weather_df = pd.DataFrame({
        '일자': dates,
        '평균기온': np.random.uniform(25.0, 34.0, size=7),
        '일교차': np.random.uniform(6.0, 13.0, size=7),
        '상대습도': np.random.uniform(55.0, 88.0, size=7)
    })
    return weather_df

# ==========================================
# 4. 위험도 산출 및 CSV 저장 메인 파이프라인
# ==========================================
def main():
    print("=== [Step 2] 천안시 순회진료 적정 일자 및 위험도 분석 시작 ===")
    
    if not os.path.exists(WEIGHTS_CSV_PATH):
        print(f"[오류] 가중치 파일({WEIGHTS_CSV_PATH})이 존재하지 않습니다.")
        return
        
    weights_df = pd.read_csv(WEIGHTS_CSV_PATH)
    print(f"[정보] 질병 가중치 모델 로드 완료 ({len(weights_df)}개 질환)")
    
    weather_df = get_weather_forecast_7days()
    
    # 1) 미래 예보 수치 매핑
    raw_pm = fetch_pm25_forecast(SERVICE_KEY)
    pm_mapping = parse_pm25_to_dict(raw_pm, target_region="충남")
    weather_df['PM25'] = weather_df['일자'].map(pm_mapping).fillna(20.0)
    
    # 2) 오늘 날짜 실시간 관측값 덮어쓰기
    today_str = datetime.now().strftime("%Y-%m-%d")
    realtime_pm25 = fetch_realtime_pm25(SERVICE_KEY, station_name="성성동")
    
    if realtime_pm25 is not None:
        weather_df.loc[weather_df['일자'] == today_str, 'PM25'] = realtime_pm25
        print(f"[반영] 오늘({today_str}) 미세먼지 수치를 실시간 관측값({realtime_pm25} ㎍/㎥)으로 갱신했습니다.")

    regions = ['성환읍', '직산읍', '성거읍', '병천면', '동면', '백석동', '신방동']
    summary_list = []
    detailed_list = []
    
    print("[정보] 일자별·지역별·질병별 위험도 스코어링 산출 중...")
    
    for idx, row in weather_df.iterrows():
        dt = row['일자']
        t_mean = row['평균기온']
        t_range = row['일교차']
        hum = row['상대습도']
        pm25 = row['PM25']
        
        for region in regions:
            region_factor = 1.0 if '읍' in region or '면' in region else 0.95
            
            regional_disease_scores = {}
            for _, w_row in weights_df.iterrows():
                dis_code = w_row['질병코드']
                dis_name = w_row.get('질병명', dis_code)
                
                w_temp = w_row.get('가중치_기온', 0.25)
                w_range = w_row.get('가중치_일교차', 0.25)
                w_hum = w_row.get('가중치_습도', 0.25)
                w_pm = w_row.get('가중치_미세먼지', 0.25)
                
                score = (
                    (t_mean / 35.0 * w_temp) + 
                    (t_range / 15.0 * w_range) + 
                    (hum / 100.0 * w_hum) + 
                    (pm25 / 100.0 * w_pm)
                ) * 100 * region_factor
                
                score = round(min(max(score, 20.0), 99.9), 1)
                regional_disease_scores[dis_name] = score
                
                detailed_list.append({
                    '일자': dt,
                    '지역': region,
                    '질병코드': dis_code,
                    '질병명': dis_name,
                    '위험도점수': score
                })
            
            sorted_diseases = sorted(regional_disease_scores.items(), key=lambda x: x[1], reverse=True)
            top_disease, top_score = sorted_diseases[0]
            
            risk_level = "고위험" if top_score >= 75 else ("주의" if top_score >= 60 else "안전")
            
            summary_list.append({
                '일자': dt,
                '지역': region,
                '평균기온': round(t_mean, 1),
                '일교차': round(t_range, 1),
                '상대습도': round(hum, 1),
                '초미세먼지(PM2.5)': round(pm25, 1),
                '종합위험도점수': top_score,
                '위험단계': risk_level,
                '최우선관심질환': f"{top_disease} ({top_score}점)"
            })
            
    summary_df = pd.DataFrame(summary_list)
    detailed_df = pd.DataFrame(detailed_list)
    
    summary_csv_name = "천안시_일자별_위험도_요약.csv"
    detailed_csv_name = "천안시_질병별_위험도_상세.csv"
    
    summary_df.to_csv(summary_csv_name, index=False, encoding='utf-8-sig')
    detailed_df.to_csv(detailed_csv_name, index=False, encoding='utf-8-sig')
    
    print("=" * 50)
    print(f"[완료] 분석 결과 파일 저장 성공!")
    print(f" 1. 요약 파일: {summary_csv_name}")
    print(f" 2. 상세 파일: {detailed_csv_name}")
    print("=" * 50)
    print(summary_df.head(5))

if __name__ == "__main__":
    main()