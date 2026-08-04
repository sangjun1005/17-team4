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

# 천안시 실제 미세먼지 측정소 목록 (6개소)
CHEONAN_STATIONS = ['성성동', '수신면', '신방동', '성황동', '성거읍', '백석동']

# 천안시 읍면동별 격자 좌표 및 실제 측정소 매핑 정보
CHEONAN_DONGS = [
    {'district': '동남구', 'dong': '목천읍', 'nx': 63, 'ny': 110, 'station': '신방동'},
    {'district': '동남구', 'dong': '병천면', 'nx': 65, 'ny': 111, 'station': '수신면'},
    {'district': '동남구', 'dong': '신방동', 'nx': 63, 'ny': 111, 'station': '신방동'},
    {'district': '동남구', 'dong': '원성1동', 'nx': 63, 'ny': 112, 'station': '성황동'},
    {'district': '동남구', 'dong': '청룡동', 'nx': 63, 'ny': 111, 'station': '신방동'},
    {'district': '서북구', 'dong': '성환읍', 'nx': 58, 'ny': 114, 'station': '성성동'},
    {'district': '서북구', 'dong': '성거읍', 'nx': 61, 'ny': 114, 'station': '성거읍'},
    {'district': '서북구', 'dong': '백석동', 'nx': 60, 'ny': 111, 'station': '백석동'},
    {'district': '서북구', 'dong': '불당동', 'nx': 60, 'ny': 110, 'station': '성성동'},
    {'district': '서북구', 'dong': '성성동', 'nx': 60, 'ny': 112, 'station': '성성동'}
]

# 7대 핵심 질병의 회귀계수(Beta) 및 기본 점수 체계
# 분석 결과(Marginal_R2 및 기상변수별 가중치 %) 기반 고도화된 질병별 민감도 모델
DISEASE_MODELS = {
    # 1. I10 본태성(원발성) 고혈압 (R2: 0.2875)
    '본태성(원발성) 고혈압': {
        'temp': -0.04,      # 기온 가중치 28.99%
        'humidity': 0.03,  # 습도 가중치 19.06%
        'pm25': 0.05,      # PM2.5 가중치 32.08%
        'diurnal': 0.03,   # 일교차 가중치 19.88%
        'base_score': 50.0
    },

    # 2. J06 다발성 및 상세불명 부위의 급성 상기도감염 (R2: 0.6177)
    '급성 상기도감염': {
        'temp': -0.17,      # 기온 가중치 56.46% (압도적 기온 민감)
        'humidity': 0.04,  # 습도 가중치 13.69%
        'pm25': 0.05,      # PM2.5 가중치 16.74%
        'diurnal': 0.04,   # 일교차 가중치 13.12%
        'base_score': 50.0
    },

    # 3. J20 급성 기관지염 (R2: 0.6040)
    '급성 기관지염': {
        'temp': -0.17,      # 기온 가중치 56.34%
        'humidity': 0.03,  # 습도 가중치 8.59%
        'pm25': 0.07,      # PM2.5 가중치 21.92%
        'diurnal': 0.04,   # 일교차 가중치 13.15%
        'base_score': 50.0
    },

    # 4. J30 혈관운동성 및 알레르기성 비염 (R2: 0.6750)
    '혈관운동성 및 알레르기성 비염': {
        'temp': -0.15,      # 기온 가중치 43.39%
        'humidity': 0.06,  # 습도 가중치 18.67%
        'pm25': 0.06,      # PM2.5 가중치 17.06%
        'diurnal': 0.07,   # 일교차 가중치 20.88%
        'base_score': 50.0
    },

    # 5. J45 천식 (R2: 0.5529)
    '천식': {
        'temp': -0.11,      # 기온 가중치 40.02%
        'humidity': 0.11,  # 습도 가중치 38.74% (높은 습도 민감성 반영)
        'pm25': 0.04,      # PM2.5 가중치 16.26%
        'diurnal': 0.01,   # 일교차 가중치 4.97%
        'base_score': 50.0
    },

    # 6. K29 위염 및 십이지장염 (R2: 0.4360)
    '위염 및 십이지장염': {
        'temp': -0.11,      # 기온 가중치 51.02%
        'humidity': 0.06,  # 습도 가중치 29.00%
        'pm25': 0.01,      # PM2.5 가중치 6.39%
        'diurnal': 0.03,   # 일교차 가중치 13.59%
        'base_score': 50.0
    },

    # 7. L23 알레르기성 접촉피부염 (R2: 0.8573) - 신규 추가
    '알레르기성 접촉피부염': {
        'temp': 0.26,       # 기온 가중치 61.10% (여름철 고온/땀 자극 반영)
        'humidity': 0.04,  # 습도 가중치 8.69%
        'pm25': 0.11,      # PM2.5 가중치 26.21%
        'diurnal': 0.02,   # 일교차 가중치 3.99%
        'base_score': 50.0
    },

    # 8. M17 무릎관절증 (R2: 0.4256)
    '무릎관절증': {
        'temp': -0.03,      # 기온 가중치 13.72%
        'humidity': 0.05,  # 습도 가중치 23.49%
        'pm25': 0.06,      # PM2.5 가중치 27.08%
        'diurnal': 0.08,   # 일교차 가중치 35.71% (높은 일교차/관절통 반영)
        'base_score': 50.0
    },

    # 9. M54 등통증 (R2: 0.5556) - 신규 추가
    '등통증': {
        'temp': -0.02,      # 기온 가중치 8.29%
        'humidity': 0.09,  # 습도 가중치 31.37%
        'pm25': 0.07,      # PM2.5 가중치 23.87%
        'diurnal': 0.10,   # 일교차 가중치 36.47% (압도적 일교차 민감성 반영)
        'base_score': 50.0
    }
}

# ==========================================
# 2. 실시간 미세먼지 수집 함수 (검증 완료 로직)
# ==========================================

def fetch_cheonan_realtime_pm(service_key, stations=CHEONAN_STATIONS):
    """에어코리아 천안시 측정소별 당일 실시간 미세먼지(PM2.5) 수치 수집 (안전 보완형)"""
    url = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
    station_pm25_map = {}

    print("📡 [API 연동] 천안시 측정소별 실시간 미세먼지 수집 중...")
    
    for station in stations:
        params = {
            'serviceKey': service_key,
            'returnType': 'xml',
            'numOfRows': '10',
            'pageNo': '1',
            'stationName': station,
            'dataTerm': 'DAILY',
            'ver': '1.0'
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                
                # API 결과 코드 확인 (정상: 00)
                result_code = root.find('.//resultCode')
                if result_code is not None and result_code.text != '00':
                    print(f"  └ [{station}] API 오류 코드: {result_code.text} -> 기본값 적용")
                    station_pm25_map[station] = 18.0
                    continue

                item = root.find('.//item')
                if item is not None:
                    pm25_val = item.find('pm25Value')
                    if pm25_val is not None and pm25_val.text:
                        # isdigit() 대신 float 변환 사용 (소수점 및 다양한 숫자 형태 완벽 대응)
                        try:
                            val = float(pm25_val.text)
                            station_pm25_map[station] = val
                            print(f"  └ [{station}] PM2.5 수집 성공: {val}")
                            continue
                        except ValueError:
                            # '-' 이거나 공백 등 숫자로 바꿀 수 없는 경우
                            pass
                            
            print(f"  └ [{station}] 데이터 없음/파싱 실패 -> 기본값(18.0) 적용")
            station_pm25_map[station] = 18.0
            
        except Exception as e:
            print(f"  └ [{station}] 요청 중 에러 발생: {e} -> 기본값(18.0) 적용")
            station_pm25_map[station] = 18.0
            
    return station_pm25_map


# ==========================================
# 2-1. 초미세먼지 주간예보(충남) 수집 + 실시간값과 결합한 5일치 추정 함수
# ==========================================

def fetch_weekly_pm25_forecast(service_key, region_name='충남', debug=False):
    """
    에어코리아 초미세먼지 주간예보(getMinuDustWeekFrcstDspth) API로
    '충남' 권역의 낮음/높음 등급을 날짜별로 수집.

    반환: {'YYYY-MM-DD': '낮음' | '높음', ...}

    주의(실측 검증됨):
    - 예보 내용 필드는 frcstOneCd가 아니라 frcstOneCn (Content) 이며, 지역별 등급이
      "서울 : 낮음,인천 : 낮음,...,충남 : 낮음,..." 형태의 콤마 구분 문자열로 들어있음.
    - 실제 제공되는 예보일수는 One~Four(4일치)까지만 확인됨.
    - searchDate에 '오늘' 날짜를 넣으면 조회가 안 되는 경우가 많아,
      어제 날짜 -> 오늘 날짜 -> 파라미터 미포함(서버 기본값) 순으로 재시도.
    """
    url = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMinuDustWeekFrcstDspth"
    grade_map = {}

    today_dt = datetime.datetime.now()
    yesterday_str = (today_dt - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    today_str = today_dt.strftime("%Y-%m-%d")

    attempt_params = [
        ("오늘-1일(어제)", {'serviceKey': service_key, 'returnType': 'xml', 'numOfRows': '100', 'pageNo': '1', 'searchDate': yesterday_str}),
        ("오늘 날짜", {'serviceKey': service_key, 'returnType': 'xml', 'numOfRows': '100', 'pageNo': '1', 'searchDate': today_str}),
        ("파라미터 미포함", {'serviceKey': service_key, 'returnType': 'xml', 'numOfRows': '100', 'pageNo': '1'}),
    ]

    day_keys = [
        ('frcstOneDt', 'frcstOneCn'),
        ('frcstTwoDt', 'frcstTwoCn'),
        ('frcstThreeDt', 'frcstThreeCn'),
        ('frcstFourDt', 'frcstFourCn'),
    ]

    region_pattern = re.compile(rf'{region_name}\s*:\s*([가-힣]+)')

    print(f"📡 [API 연동] 초미세먼지 주간예보({region_name}) 수집 시작 (우선 조회일: {yesterday_str})...")

    for label, params in attempt_params:
        try:
            response = requests.get(url, params=params, timeout=10)
            if debug:
                print(response.text[:2000])

            if response.status_code != 200:
                print(f"  └ [{label}] 응답 오류 (status={response.status_code}) -> 다음 방법으로 시도")
                continue

            root = ET.fromstring(response.content)
            result_code = root.find('.//resultCode')
            if result_code is not None and result_code.text != '00':
                print(f"  └ [{label}] API 오류 코드: {result_code.text} -> 다음 방법으로 시도")
                continue

            items = root.findall('.//item')
            if not items:
                print(f"  └ [{label}] 조회 결과 0건 -> 다음 방법으로 시도")
                continue

            for item in items:
                for dt_tag, cn_tag in day_keys:
                    forecast_date = item.findtext(dt_tag, '')
                    forecast_cn = item.findtext(cn_tag, '')
                    if not forecast_date or not forecast_cn:
                        continue

                    match = region_pattern.search(forecast_cn)
                    if not match:
                        continue
                    grade = match.group(1)

                    # 날짜 형식이 YYYYMMDD로 올 수도 있어 정규화
                    norm_date = forecast_date.replace('.', '-').replace('/', '-')
                    if re.fullmatch(r'\d{8}', norm_date):
                        norm_date = f"{norm_date[0:4]}-{norm_date[4:6]}-{norm_date[6:8]}"

                    if grade in ('낮음', '높음'):
                        grade_map[norm_date] = grade

            if grade_map:
                print(f"  └ [{label}] 기준 {region_name} 예보 {len(grade_map)}일치 수집 성공!")
                for d, g in sorted(grade_map.items()):
                    print(f"     · {d}: {g}")
                break
            else:
                print(f"  └ [{label}] 항목은 있으나 {region_name} 등급 파싱 실패 -> 다음 방법으로 시도")

        except Exception as e:
            print(f"  └ [{label}] 요청 중 예외 발생: {e} -> 다음 방법으로 시도")

    if not grade_map:
        print(f"  └ 모든 시도 실패 -> {region_name} 주간예보 없이 진행(지속성 가정 적용)")

    return grade_map


def estimate_future_pm25(today_actual_pm25, region_today_anchor, grade, day_offset,
                          decay_rate=0.6, low_anchor=20.0, high_anchor=45.0):
    """
    오늘 실측 PM2.5(측정소 단위, 고해상도)와 충남 권역 주간예보 등급(낮음/높음, 저해상도)을
    결합해 향후 D+day_offset일의 PM2.5를 신뢰도 있게 추정.

    - grade가 없으면(예보 미제공) 측정소의 오늘 편차를 감쇠시키며 유지(지속성 가정)
    - grade가 있으면 등급별 대표값(anchor)을 중심으로, 측정소가 평소 지역평균 대비
      높거나 낮은 경향(offset)을 day_offset에 따라 감쇠 반영
    """
    # 오늘 이 측정소가 지역 평균(anchor) 대비 얼마나 높거나 낮은 경향인지
    station_offset = today_actual_pm25 - region_today_anchor
    decay = decay_rate ** day_offset

    if grade == '낮음':
        anchor = low_anchor
        low_bound, high_bound = 0.0, 35.0
    elif grade == '높음':
        anchor = high_anchor
        low_bound, high_bound = 36.0, 150.0
    else:
        # 예보 등급을 못 가져온 경우: 오늘 실측치를 서서히 기준값(18.0)으로 회귀시키는 지속성 모델
        anchor = 18.0
        low_bound, high_bound = 0.0, 150.0

    estimated = anchor + (station_offset * decay)
    estimated = max(low_bound, min(high_bound, estimated))
    return round(float(estimated), 1)


# ==========================================
# 3. 비선형 감쇄 스케일링을 통한 100점 만점 제어 함수
# ==========================================

def calculate_disease_risks(obs_temp, obs_hum, obs_pm25, obs_diurnal):
    """
    100점 만점 구조를 유지하되, 극단적 최고 점수가 너무 자주 나오지 않도록
    비선형 함수(Hyperbolic Tangent / Saturation Scaling)를 적용한 위험도 산출
    """
    scores = {}
    
    base_temp, base_hum, base_pm25, base_diurnal = 25.5, 75.0, 18.0, 8.5
    
    delta_temp = obs_temp - base_temp
    delta_hum = obs_hum - base_hum
    delta_pm25 = obs_pm25 - base_pm25
    delta_diurnal = obs_diurnal - base_diurnal
    
    for disease, betas in DISEASE_MODELS.items():
        impact = (
            (delta_temp * betas['temp']) +
            (delta_hum * betas['humidity']) +
            (delta_pm25 * betas['pm25']) +
            (delta_diurnal * betas['diurnal'])
        )
        
        sensitivity_weight = sum([abs(v) for k, v in betas.items() if k != 'base_score'])
        
        # 선형 변화량 계산
        linear_shift = impact * sensitivity_weight * 20.0
        
        # 💡 [핵심 비선형 감쇄 적용]
        # 점수가 50점을 기준으로 위아래로 움직이되, 100점(또는 0점) 근처로 갈수록 저항이 생겨 
        # 무한정 치솟지 않고 부드럽게 수렴(Saturation)하도록 탄젠트(tanh) 변환 활용
        # 공식: 50 + 50 * tanh(linear_shift / 35.0)
        # 이렇게 하면 평상시에는 40~60점대에서 유연하게 움직이고, 기상이 엄청나게 나빠져야만 85~95점대로 진입하며,
        # 이론상 최대 100점 만점은 열려있되 정말 극단적인 재난급 날씨가 아니면 도달 불가능해집니다.
        saturated_score = 50.0 + 50.0 * np.tanh(linear_shift / 35.0)
        
        # 안전하게 10.0 ~ 100.0 범위 클리핑
        final_score = max(10.0, min(100.0, saturated_score))
        scores[disease] = round(float(final_score), 1)
        
    return scores


# ==========================================
# 4. 메인 파이프라인 및 실행
# ==========================================

def main():
    print("🚀 [Step2_r8] 비선형 감쇄 100점 만점 제어 + 실시간/주간예보 결합 미세먼지 파이프라인 실행 중...")
    
    realtime_pm_map = fetch_cheonan_realtime_pm(AIR_SERVICE_KEY, CHEONAN_STATIONS)
    weekly_grade_map = fetch_weekly_pm25_forecast(AIR_SERVICE_KEY, region_name='충남')

    # 오늘의 충남(천안) 측정소 평균 = 향후 예보 등급을 숫자로 환산할 때 기준이 되는 지역 평균 앵커
    region_today_anchor = float(np.mean(list(realtime_pm_map.values()))) if realtime_pm_map else 18.0

    base_date = datetime.datetime.now()
    dates = [(base_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]

    # 날짜별 PM2.5 추정에 사용할 예보 등급 로그 (없으면 마지막으로 확인된 등급을 지속성 가정으로 유지)
    last_known_grade = None
    date_grade_for_log = {}
    for day_offset, date_str in enumerate(dates):
        if day_offset == 0:
            continue
        grade = weekly_grade_map.get(date_str)
        if grade is None:
            grade = last_known_grade  # 예보 미제공일은 직전 알려진 등급을 지속
        else:
            last_known_grade = grade
        date_grade_for_log[date_str] = grade

    previous_scores_memory = {}
    summary_results = []
    detail_results = []
    
    for day_offset, date_str in enumerate(dates):
        for loc in CHEONAN_DONGS:
            district = loc['district']
            dong = loc['dong']
            station = loc['station']

            station_today_actual = realtime_pm_map.get(station, 18.0)

            if day_offset == 0:
                # 오늘 = 실시간 측정값 그대로 사용
                daily_pm25 = station_today_actual
                pm25_grade = '실시간'
                pm25_confidence = 'high'
            else:
                grade = date_grade_for_log.get(date_str)
                daily_pm25 = estimate_future_pm25(
                    today_actual_pm25=station_today_actual,
                    region_today_anchor=region_today_anchor,
                    grade=grade,
                    day_offset=day_offset
                )
                pm25_grade = grade if grade else '예보없음(지속성가정)'
                # 예보 등급이 있으면 confidence는 day_offset이 커질수록 감소
                pm25_confidence = 'medium' if grade else 'low'

            simulated_temp = 25.5 + ((hash(date_str + dong) % 9) - 4)
            simulated_hum = 75.0 + ((hash(dong + date_str) % 30) - 15)
            simulated_diurnal = 8.5 + ((hash(date_str) % 7) - 3)
            
            current_disease_scores = calculate_disease_risks(
                simulated_temp, simulated_hum, daily_pm25, simulated_diurnal
            )
            
            # 시계열 스무딩 (전일 점수 40% + 당일 점수 60%)
            loc_key = f"{district}_{dong}"
            smoothed_scores = {}
            if loc_key in previous_scores_memory:
                prev_scores = previous_scores_memory[loc_key]
                for d_name, cur_val in current_disease_scores.items():
                    prev_val = prev_scores.get(d_name, cur_val)
                    smoothed_val = (0.4 * prev_val) + (0.6 * cur_val)
                    smoothed_scores[d_name] = round(float(smoothed_val), 1)
            else:
                smoothed_scores = current_disease_scores
                
            previous_scores_memory[loc_key] = smoothed_scores
            
            score_values = list(smoothed_scores.values())
            mean_risk = np.mean(score_values)
            max_risk = np.max(score_values)
            
            # Max-Driven 하이브리드 Total Risk
            total_risk = round((0.5 * mean_risk) + (0.5 * max_risk), 1)
            
            # Red Flag 경고 체계 (75점 이상 치솟는 질환 감지)
            WARNING_THRESHOLD = 75.0
            high_risk = []
            for d_name, score in smoothed_scores.items():
                if score >= WARNING_THRESHOLD:
                    high_risk.append(f"{d_name}({score}점)")
                    
            is_warning = len(high_risk) > 0
            warning_str = ", ".join(high_risk) if is_warning else "안전"
            
            summary_results.append({
                'date': date_str,
                'district': district,
                'dong': dong,
                'station': station,
                'total_risk': total_risk,
                'mean_risk': round(mean_risk, 1),
                'max_risk': round(max_risk, 1),
                'is_warning': is_warning,
                'high_risk_diseases': warning_str,
                'temp': simulated_temp,
                'humidity': simulated_hum,
                'pm25_actual': daily_pm25,
                'pm25_source': pm25_grade,
                'pm25_confidence': pm25_confidence
            })
            
            detail_row = {
                'date': date_str,
                'district': district,
                'dong': dong,
                'station': station,
                'pm25_actual': daily_pm25,
                'pm25_source': pm25_grade,
                'pm25_confidence': pm25_confidence,
                'total_risk': total_risk
            }
            detail_row.update(smoothed_scores)
            detail_results.append(detail_row)

    df_summary = pd.DataFrame(summary_results)
    df_detail = pd.DataFrame(detail_results)
    
    df_summary = df_summary.sort_values(by=['is_warning', 'total_risk'], ascending=[False, False])

    today_file_str = datetime.datetime.now().strftime("%Y%m%d")
    summary_csv = f"천안시_일자별_위험도_요약_r8_{today_file_str}.csv"
    detail_csv = f"천안시_질병별_위험도_상세_r8_{today_file_str}.csv"

    df_summary.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    df_detail.to_csv(detail_csv, index=False, encoding='utf-8-sig')

    print("\n==========================================================================================")
    print(f"💾 [100점 만점 / 비선형 감쇄 적용 완료] 요약 CSV -> {summary_csv}")
    print(f"💾 [100점 만점 / 비선형 감쇄 적용 완료] 상세 CSV -> {detail_csv}")
    print("==========================================================================================")
    
    print("\n📊 [순회진료 추천 상위 5개 지역 및 일자]")
    print(df_summary[['date', 'district', 'dong', 'total_risk', 'high_risk_diseases']].head(5).to_string(index=False))
    print("==========================================================================================")


if __name__ == "__main__":
    main()