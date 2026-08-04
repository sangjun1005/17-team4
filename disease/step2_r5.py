import os
import re
import time
import datetime
import requests
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

# ==========================================
# 1. 설정 및 기본 데이터 정의
# ==========================================

KMA_AUTH_KEY = "8wUF4lmCQuGFBeJZggLhhQ"  # 기상청 API 인증키
AIR_SERVICE_KEY = "6069cd5378ffe531429bdca0ff28ff0f0b0e661007504bc6270cc80995a053b9"  # 에어코리아 API 키

CHEONAN_DONGS = [
    {'district': '동남구', 'dong': '목천읍', 'nx': 63, 'ny': 110, 'station': '신방동'},
    {'district': '동남구', 'dong': '병천면', 'nx': 65, 'ny': 111, 'station': '병천면'},
    {'district': '동남구', 'dong': '신방동', 'nx': 63, 'ny': 111, 'station': '신방동'},
    {'district': '동남구', 'dong': '원성1동', 'nx': 63, 'ny': 112, 'station': '신방동'},
    {'district': '동남구', 'dong': '청룡동', 'nx': 64, 'ny': 111, 'station': '신방동'},
    {'district': '서북구', 'dong': '성환읍', 'nx': 60, 'ny': 117, 'station': '성성동'},
    {'district': '서북구', 'dong': '성거읍', 'nx': 62, 'ny': 116, 'station': '성성동'},
    {'district': '서북구', 'dong': '직산읍', 'nx': 61, 'ny': 115, 'station': '성성동'},
    {'district': '서북구', 'dong': '부성1동', 'nx': 61, 'ny': 113, 'station': '성성동'},
    {'district': '서북구', 'dong': '백석동', 'nx': 61, 'ny': 112, 'station': '성성동'}
]

# 질병별 모델 파라미터 (회귀계수 및 기본 점수)
DISEASE_MODELS = {
    '본태성(원발성) 고혈압': {'temp': -0.05, 'humidity': 0.02, 'pm25': 0.04, 'diurnal': 0.15, 'base_score': 50.0},
    '혈관운동성 및 알레르기성 비염': {'temp': -0.10, 'humidity': 0.15, 'pm25': 0.08, 'diurnal': 0.12, 'base_score': 50.0},
    '위염 및 십이지장염': {'temp': -0.03, 'humidity': 0.05, 'pm25': 0.03, 'diurnal': 0.05, 'base_score': 50.0},
    '천식': {'temp': -0.15, 'humidity': 0.08, 'pm25': 0.12, 'diurnal': 0.10, 'base_score': 50.0},
    '급성 상기도감염': {'temp': -0.12, 'humidity': 0.10, 'pm25': 0.10, 'diurnal': 0.14, 'base_score': 50.0},
    '무릎관절증': {'temp': -0.20, 'humidity': 0.18, 'pm25': 0.02, 'diurnal': 0.25, 'base_score': 50.0},
    '알레르기성 접촉피부염': {'temp': -0.05, 'humidity': 0.20, 'pm25': 0.05, 'diurnal': 0.08, 'base_score': 50.0}
}


# ==========================================
# 2. 기상 및 대기오염 데이터 수집 함수
# ==========================================

def get_kma_forecast(nx, ny):
    """기상청 단기예보 API를 활용하여 향후 기상 데이터 수집"""
    base_date = datetime.datetime.now().strftime("%Y%m%d")
    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params = {
        'serviceKey': KMA_AUTH_KEY,
        'pageNo': '1',
        'numOfRows': '300',
        'dataType': 'XML',
        'base_date': base_date,
        'base_time': '0500',
        'nx': nx,
        'ny': ny
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return {}
        
        root = ET.fromstring(response.content)
        items = root.findall('.//item')
        
        forecast_data = {}
        for item in items:
            fcst_date = item.find('fcstDate').text
            fcst_time = item.find('fcstTime').text
            category = item.find('category').text
            fcst_value = item.find('fcstValue').text
            
            date_str = f"{fcst_date[:4]}-{fcst_date[4:6]}-{fcst_date[6:]}"
            if date_str not in forecast_data:
                forecast_data[date_str] = {'temps': [], 'hums': []}
                
            if category == 'TMP': # 기온
                forecast_data[date_str]['temps'].append(float(fcst_value))
            elif category == 'REH': # 습도
                forecast_data[date_str]['hums'].append(float(fcst_value))
                
        # 일자별 평균 기온, 일교차, 평균 습도 계산
        daily_weather = {}
        for date_str, data in forecast_data.items():
            temps = data['temps']
            hums = data['hums']
            if temps:
                t_max = max(temps)
                t_min = min(temps)
                t_avg = np.mean(temps)
                diurnal = t_max - t_min
                hum_avg = np.mean(hums) if hums else 60.0
                daily_weather[date_str] = {
                    'temp': t_avg,
                    'diurnal': diurnal,
                    'humidity': hum_avg
                }
        return daily_weather
    except Exception as e:
        print(f"기상청 API 연동 오류 (nx={nx}, ny={ny}): {e}")
        return {}


def get_air_pollution(station_name):
    """에어코리아 실시간 측정소별 미세먼지 정보 조회"""
    url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
    params = {
        'serviceKey': AIR_SERVICE_KEY,
        'returnType': 'xml',
        'numOfRows': '10',
        'pageNo': '1',
        'stationName': station_name,
        'dataTerm': 'DAILY',
        'ver': '1.0'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return 30.0 # 기본값
        
        root = ET.fromstring(response.content)
        item = root.find('.//item')
        if item is not None and item.find('pm25Value') is not None:
            val = item.find('pm25Value').text
            if val and val.isdigit():
                return float(val)
        return 30.0
    except Exception:
        return 30.0


# ==========================================
# 3. 위험도 스코어링 및 메인 실행 로직
# ==========================================

def main():
    print("🚀 천안시 기상 연동 질병 위험도 분석 및 CSV 저장 작업을 시작합니다...")
    
    summary_results = []
    detail_results = []
    
    # 오늘부터 향후 5일간 날짜 설정
    today = datetime.datetime.now()
    target_dates = [(today + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
    
    for loc in CHEONAN_DONGS:
        district = loc['district']
        dong = loc['dong']
        nx = loc['nx']
        ny = loc['ny']
        station = loc['station']
        
        print(f"[{district} {dong}] 기상 및 대기오염 데이터 수집 중...")
        weather_data = get_kma_forecast(nx, ny)
        pm25_val = get_air_pollution(station)
        
        for date_str in target_dates:
            w_info = weather_data.get(date_str, {'temp': 25.0, 'diurnal': 10.0, 'humidity': 60.0})
            daily_temp = w_info['temp']
            daily_diurnal = w_info['diurnal']
            daily_hum = w_info['humidity']
            daily_pm25 = pm25_val
            
            # 질병별 위험도 계산
            disease_scores = {}
            for dis_name, model in DISEASE_MODELS.items():
                # 간단한 선형 결합 스코어링 공식 적용
                score = model['base_score'] + \
                        (daily_temp * model['temp'] * 2) + \
                        (daily_diurnal * model['diurnal'] * 1.5) + \
                        (daily_hum * model['humidity'] * 0.1) + \
                        (daily_pm25 * model['pm25'] * 0.5)
                score = np.clip(score, 0, 100)
                disease_scores[dis_name] = round(score, 1)
            
            # 종합 위험도 (평균) 산출
            total_risk = round(np.mean(list(disease_scores.values())), 1)
            
            # 주의/고위험 질환 추출 (예: 60점 이상)
            warning_diseases = [k for k, v in disease_scores.items() if v >= 60.0]
            warning_str = ", ".join(warning_diseases) if warning_diseases else "안전"
            
            # 요약 데이터 행 구성 (기상 정보 포함)
            summary_results.append({
                '날짜': date_str,
                '행정구': district,
                '읍면동': dong,
                '측정소': station,
                '평균기온(℃)': round(daily_temp, 1),
                '일교차(℃)': round(daily_diurnal, 1),
                '습도(%)': round(daily_hum, 1),
                '초미세먼지(㎍/㎥)': round(daily_pm25, 1),
                '종합위험도': total_risk,
                '주의질환목록': warning_str
            })
            
            # 상세 데이터 행 구성 (기상 정보 + 질병별 점수 포함)
            detail_row = {
                '날짜': date_str,
                '행정구': district,
                '읍면동': dong,
                '측정소': station,
                '평균기온(℃)': round(daily_temp, 1),
                '일교차(℃)': round(daily_diurnal, 1),
                '습도(%)': round(daily_hum, 1),
                '초미세먼지(㎍/㎥)': round(daily_pm25, 1),
                '종합위험도': total_risk
            }
            detail_row.update(disease_scores)
            detail_results.append(detail_row)
            
        time.sleep(0.2) # API 부하 방지용 딜레이
        
    df_summary = pd.DataFrame(summary_results)
    df_detail = pd.DataFrame(detail_results)
    
    # 종합 점수(종합위험도) 기준 내림차순 정렬
    if not df_summary.empty:
        df_summary = df_summary.sort_values(by=['종합위험도'], ascending=False)
        df_detail = df_detail.sort_values(by=['종합위험도'], ascending=False)

    today_file_str = datetime.datetime.now().strftime("%Y%m%d")
    summary_csv = f"천안시_일자별_위험도_요약_기상포함_{today_file_str}.csv"
    detail_csv = f"천안시_질병별_위험도_상세_기상포함_{today_file_str}.csv"

    # CSV 파일로 저장 (엑셀 한글 깨짐 방지 utf-8-sig 인코딩 적용)
    df_summary.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    df_detail.to_csv(detail_csv, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print("✨ 분석 및 CSV 파일 저장 완료!")
    print(f"📁 1. 요약 파일: {summary_csv}")
    print(f"📁 2. 상세 파일: {detail_csv}")
    print("="*50)

if __name__ == "__main__":
    main()