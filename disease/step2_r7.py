import os
import re
import time
import hashlib
import datetime
import requests
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

# ==========================================
# 0. 점수 체계 튜닝 상수 (실제 운영 데이터로 추후 보정 필요)
# ==========================================

LINEAR_SHIFT_SCALE = 20.0       # delta*beta 합(impact)을 얼마나 증폭할지. 과거엔 sensitivity_weight까지 중복곱해 과증폭됐던 부분을 제거
TANH_SATURATION_DIVISOR = 35.0  # tanh 포화 속도. 값이 작을수록 극단치에 더 쉽게 도달함 (임의값, 백테스트 필요)
SCORE_FLOOR = 10.0
SCORE_CEIL = 100.0
WARNING_THRESHOLD = 75.0        # Red Flag 경고 임계값 (임의값, 백테스트 필요)

# 시계열 스무딩 시 '오늘(당일) 값'에 부여할 가중치. pm25_confidence에 따라 다르게 적용.
# high(실측) -> 현재값을 많이 신뢰 / medium(주간예보 추정) -> 절반 정도만 반영 / low(예보없음, 지속성가정) -> 과거값 위주로 유지
CONFIDENCE_CURRENT_WEIGHT = {
    'high': 0.60,
    'medium': 0.45,
    'low': 0.30,
}


def _stable_hash(text: str) -> int:
    """
    파이썬 내장 hash()는 프로세스마다 시드가 랜덤(PYTHONHASHSEED)이라 같은 문자열도
    실행할 때마다 다른 값을 반환함. 동일 입력 -> 항상 동일 출력을 보장하는
    결정론적 해시로 대체 (날짜/동네별 시뮬레이션 기온·습도·일교차의 재현성 확보용).
    """
    return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)


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

    ⚠️ 실측으로 확인된 API 동작 방식(중요):
    - searchDate 파라미터 없이 호출하면 예보 '내용'이 아니라 최근 발표일자(presnatnDt) 목록만 돌아옴.
    - 실제 예보 내용(frcstOneCn 등)을 받으려면, 그 목록에 있는 '실제 발표된 날짜'를 searchDate에
      정확히 넣어서 다시 조회해야 함. '오늘'이나 '오늘-1일'이 항상 그 목록에 있다는 보장이 없음
      (예: 오늘 날짜엔 아직 발표가 없을 수 있음).
    - 서버가 가끔 SERVICETIMEOUT_ERROR(504)로 일시 실패하는 경우가 있어 재시도 로직 필요.

    따라서 2단계로 조회:
      1단계) searchDate 없이 호출 -> presnatnDt 목록 확보 -> 최신순 정렬
      2단계) 최신 발표일자부터 순서대로 searchDate에 넣어 상세 조회 (각 날짜당 최대 2회 재시도)
    """
    url = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMinuDustWeekFrcstDspth"
    grade_map = {}

    day_keys = [
        ('frcstOneDt', 'frcstOneCn'),
        ('frcstTwoDt', 'frcstTwoCn'),
        ('frcstThreeDt', 'frcstThreeCn'),
        ('frcstFourDt', 'frcstFourCn'),
    ]
    region_pattern = re.compile(rf'{region_name}\s*:\s*([^\s,]+)')

    def _check_error_and_get_items(root, label):
        """cmmMsgHeader 표준 에러 / resultCode 에러를 확인하고, 정상이면 item 리스트 반환. 실패 시 None."""
        err_header = root.find('.//cmmMsgHeader')
        if err_header is not None:
            err_code = err_header.findtext('returnReasonCode', '')
            err_msg = err_header.findtext('errMsg', '') or err_header.findtext('returnAuthMsg', '')
            print(f"  └ [{label}] ⚠️ 공공데이터포털 에러: [{err_code}] {err_msg}")
            return None

        result_code = root.find('.//resultCode')
        if result_code is not None and result_code.text != '00':
            result_msg = root.findtext('.//resultMsg', '')
            print(f"  └ [{label}] API 오류 코드: {result_code.text} ({result_msg})")
            return None

        return root.findall('.//item')

    print(f"📡 [API 연동] 초미세먼지 주간예보({region_name}) 수집 시작...")

    # ---------- 1단계: searchDate 없이 호출 -> 최근 발표일자 목록 확보 ----------
    list_params = {'serviceKey': service_key, 'returnType': 'xml', 'numOfRows': '100', 'pageNo': '1'}
    presnatn_dates = []
    try:
        response = requests.get(url, params=list_params, timeout=10)
        root = ET.fromstring(response.content)
        items = _check_error_and_get_items(root, "1단계:발표일자 목록조회")
        if items:
            presnatn_dates = sorted(
                {it.findtext('presnatnDt', '') for it in items if it.findtext('presnatnDt', '')},
                reverse=True
            )
            print(f"  └ [1단계] 최근 발표일자 {len(presnatn_dates)}건 확인: {presnatn_dates[:5]}")
    except Exception as e:
        print(f"  └ [1단계] 발표일자 목록 조회 중 예외: {e}")

    if not presnatn_dates:
        # 목록 조회 자체가 실패하면 오늘/어제로라도 최후 시도
        today_dt = datetime.datetime.now()
        presnatn_dates = [
            today_dt.strftime("%Y-%m-%d"),
            (today_dt - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
        ]
        print(f"  └ [1단계] 목록 확보 실패 -> 오늘/어제 날짜로 최후 시도: {presnatn_dates}")

    # ---------- 2단계: 최신 발표일자부터 순서대로 상세 조회 (날짜당 최대 2회 재시도) ----------
    for search_date in presnatn_dates[:3]:  # 최신 3개까지만 시도
        detail_params = {
            'serviceKey': service_key, 'returnType': 'xml',
            'numOfRows': '100', 'pageNo': '1', 'searchDate': search_date
        }

        for retry in range(2):
            label = f"2단계:{search_date}" + (f"(재시도{retry})" if retry else "")
            try:
                response = requests.get(url, params=detail_params, timeout=10)

                if response.status_code != 200:
                    print(f"  └ [{label}] HTTP 오류 (status={response.status_code}) -> 재시도")
                    time.sleep(1)
                    continue

                root = ET.fromstring(response.content)
                items = _check_error_and_get_items(root, label)
                if items is None:
                    # SERVICETIMEOUT_ERROR 등 일시적 오류일 수 있으니 재시도
                    time.sleep(1)
                    continue
                if not items:
                    print(f"  └ [{label}] 조회 결과 0건")
                    break  # 이 날짜는 재시도해도 소용없음 -> 다음 발표일자로

                if debug:
                    print(f"     [원문 일부] {response.text[:1500]}")

                for item in items:
                    for dt_tag, cn_tag in day_keys:
                        forecast_date = item.findtext(dt_tag, '')
                        forecast_cn = item.findtext(cn_tag, '')
                        if not forecast_date or not forecast_cn:
                            continue
                        match = region_pattern.search(forecast_cn)
                        if not match:
                            continue
                        grade = match.group(1).rstrip('.,')
                        norm_date = forecast_date.replace('.', '-').replace('/', '-')
                        if re.fullmatch(r'\d{8}', norm_date):
                            norm_date = f"{norm_date[0:4]}-{norm_date[4:6]}-{norm_date[6:8]}"
                        if grade in ('낮음', '높음'):
                            grade_map[norm_date] = grade

                if grade_map:
                    print(f"  └ [{label}] {region_name} 예보 {len(grade_map)}일치 수집 성공!")
                    for d, g in sorted(grade_map.items()):
                        print(f"     · {d}: {g}")
                else:
                    print(f"  └ [{label}] item은 있으나 {region_name} 패턴 매칭 실패")
                    print(f"     [원문 일부] {response.text[:1500]}")
                break  # 이 날짜에 대한 재시도 루프는 종료 (성공/실패 여부 무관, 응답은 정상 수신함)

            except ET.ParseError as e:
                print(f"  └ [{label}] XML 파싱 실패: {e}")
                break
            except Exception as e:
                print(f"  └ [{label}] 요청 중 예외 발생: {e} -> 재시도")
                time.sleep(1)

        if grade_map:
            break  # 이미 성공했으면 다음 발표일자는 시도할 필요 없음

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
# 2-2. 기상청 단기예보(기온/습도) + 중기예보(기온) 수집 함수
# ==========================================

# 중기예보(getMidTa) regId: 천안 전용 코드를 100% 확신할 수 없어, 검증된 대전(충남권 대표) 코드로 설정.
# 실제 발표문에서 지역명이 이상하면(예: 대전 기온이 나온다면) data.kma.go.kr에서 천안 전용 코드로 교체 필요.
KMA_MID_TA_REG_ID = "11C20401"  # 대전 (대전·세종·충남권 대표) — ⚠️ 검증 필요


def _get_latest_kma_issue(issue_times, buffer_minutes=10):
    """
    기상청 API는 정해진 발표시각에만 자료가 갱신됨. 현재 시각 기준으로
    이미 발표됐을 가장 최근 발표시각(base_date, base_time)을 계산.
    buffer_minutes: 발표 직후 서버 반영 지연을 감안한 여유시간.
    """
    now = datetime.datetime.now() - datetime.timedelta(minutes=buffer_minutes)
    sorted_times = sorted(issue_times)
    chosen = None
    for t in sorted_times:
        hh, mm = int(t[:2]), int(t[2:])
        if (now.hour, now.minute) >= (hh, mm):
            chosen = t
    if chosen is None:
        base_date = (now - datetime.timedelta(days=1)).strftime("%Y%m%d")
        chosen = sorted_times[-1]
    else:
        base_date = now.strftime("%Y%m%d")
    return base_date, chosen


def _check_kma_error(root, label):
    """공공데이터포털 표준 에러 포맷과 resultCode 오류를 확인. 정상이면 item 리스트, 실패면 None."""
    err_header = root.find('.//cmmMsgHeader')
    if err_header is not None:
        err_code = err_header.findtext('returnReasonCode', '')
        err_msg = err_header.findtext('errMsg', '') or err_header.findtext('returnAuthMsg', '')
        print(f"  └ [{label}] ⚠️ 공공데이터포털 에러: [{err_code}] {err_msg}")
        return None

    result_code = root.find('.//resultCode')
    if result_code is not None and result_code.text != '00':
        result_msg = root.findtext('.//resultMsg', '')
        print(f"  └ [{label}] API 오류 코드: {result_code.text} ({result_msg})")
        return None

    return root.findall('.//item')


def fetch_kma_short_forecast(nx, ny, service_key, debug=False):
    """
    기상청 단기예보(getVilageFcst)로 특정 격자(nx,ny)의 날짜별 평균기온/평균습도/일교차 수집.
    보통 발표시각 기준 최대 2~3일치(오늘~모레) 정도만 완전한 데이터가 나옴.

    반환: {'YYYY-MM-DD': {'temp': float, 'humidity': float, 'diurnal': float|None}, ...}
    """
    # ⚠️ [수정] KMA_AUTH_KEY는 공공데이터포털(apis.data.go.kr)이 아니라
    # 기상청 자체 API허브(apihub.kma.go.kr) 전용 키. 도메인과 파라미터명(authKey)이 다름.
    url = "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst"
    issue_times = ['0200', '0500', '0800', '1100', '1400', '1700', '2000', '2300']
    base_date, base_time = _get_latest_kma_issue(issue_times)

    params = {
        'authKey': service_key, 'pageNo': '1', 'numOfRows': '1000',
        'dataType': 'XML', 'base_date': base_date, 'base_time': base_time,
        'nx': str(nx), 'ny': str(ny)
    }

    result = {}
    label = f"단기예보(nx={nx},ny={ny})"
    try:
        response = requests.get(url, params=params, timeout=10)
        root = ET.fromstring(response.content)
        items = _check_kma_error(root, label)
        if not items:
            return result

        if debug:
            print(f"     [원문 일부] {response.text[:1500]}")

        daily = {}
        for item in items:
            category = item.findtext('category', '')
            fcst_date = item.findtext('fcstDate', '')
            fcst_value = item.findtext('fcstValue', '')
            if not fcst_date or fcst_value in ('', None):
                continue
            norm_date = f"{fcst_date[0:4]}-{fcst_date[4:6]}-{fcst_date[6:8]}"
            bucket = daily.setdefault(norm_date, {'TMP': [], 'REH': [], 'TMN': None, 'TMX': None})
            try:
                val = float(fcst_value)
            except ValueError:
                continue
            if category == 'TMP':
                bucket['TMP'].append(val)
            elif category == 'REH':
                bucket['REH'].append(val)
            elif category == 'TMN':
                bucket['TMN'] = val
            elif category == 'TMX':
                bucket['TMX'] = val

        for date_str, b in daily.items():
            if not b['TMP'] or not b['REH']:
                continue
            avg_temp = float(np.mean(b['TMP']))
            avg_hum = float(np.mean(b['REH']))
            if b['TMN'] is not None and b['TMX'] is not None:
                diurnal = b['TMX'] - b['TMN']
            elif len(b['TMP']) >= 2:
                diurnal = max(b['TMP']) - min(b['TMP'])
            else:
                diurnal = None
            result[date_str] = {
                'temp': round(avg_temp, 1),
                'humidity': round(avg_hum, 1),
                'diurnal': round(diurnal, 1) if diurnal is not None else None
            }

        if result:
            print(f"  └ [{label}] {len(result)}일치 수집 성공: {sorted(result.keys())}")
        else:
            print(f"  └ [{label}] item은 있으나 TMP/REH 데이터 부족")

    except Exception as e:
        print(f"  └ [{label}] 요청 중 예외 발생: {e}")

    return result


def fetch_kma_mid_temp(reg_id, service_key, debug=False):
    """
    기상청 중기예보 기온조회(getMidTa)로 D+3~D+10 최고/최저기온 수집 (습도는 미제공).

    반환: {'YYYY-MM-DD': {'temp': float(평균), 'diurnal': float, 'tmin': float, 'tmax': float}, ...}
    """
    url = "https://apihub.kma.go.kr/api/typ02/openApi/MidFcstInfoService/getMidTa"
    issue_times = ['0600', '1800']
    base_date, base_time = _get_latest_kma_issue(issue_times)
    tm_fc = f"{base_date}{base_time}"

    params = {
        'authKey': service_key, 'pageNo': '1', 'numOfRows': '10',
        'dataType': 'XML', 'regId': reg_id, 'tmFc': tm_fc
    }

    result = {}
    label = f"중기예보(regId={reg_id},tmFc={tm_fc})"
    try:
        response = requests.get(url, params=params, timeout=10)
        root = ET.fromstring(response.content)
        items = _check_kma_error(root, label)
        if not items:
            return result

        if debug:
            print(f"     [원문 일부] {response.text[:1500]}")

        item = items[0]
        issue_date = datetime.datetime.strptime(base_date, "%Y%m%d")
        for n in range(3, 11):
            tmin_text = item.findtext(f'taMin{n}', '')
            tmax_text = item.findtext(f'taMax{n}', '')
            if tmin_text == '' or tmax_text == '':
                continue
            try:
                tmin_v, tmax_v = float(tmin_text), float(tmax_text)
            except ValueError:
                continue
            target_date = (issue_date + datetime.timedelta(days=n)).strftime("%Y-%m-%d")
            result[target_date] = {
                'temp': round((tmin_v + tmax_v) / 2, 1),
                'diurnal': round(tmax_v - tmin_v, 1),
                'tmin': tmin_v,
                'tmax': tmax_v
            }

        if result:
            print(f"  └ [{label}] {len(result)}일치 수집 성공: {sorted(result.keys())}")
        else:
            print(f"  └ [{label}] item은 있으나 taMinN/taMaxN 필드 없음")

    except Exception as e:
        print(f"  └ [{label}] 요청 중 예외 발생: {e}")

    return result


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
        # impact = delta * beta 의 선형결합. 베타 자체가 이미 질병별 민감도(회귀계수)를 담고 있음.
        impact = (
            (delta_temp * betas['temp']) +
            (delta_hum * betas['humidity']) +
            (delta_pm25 * betas['pm25']) +
            (delta_diurnal * betas['diurnal'])
        )

        # ⚠️ [수정] 과거엔 여기서 sensitivity_weight(베타 절댓값 합)를 한 번 더 곱해
        # 베타가 사실상 이중으로 반영되는 버그가 있었음 (민감도가 높은 질병일수록
        # 의도한 것보다 훨씬 더 크게 증폭되는 문제). impact에 이미 베타가 반영돼 있으므로
        # 여기서는 단순 스케일링만 적용.
        linear_shift = impact * LINEAR_SHIFT_SCALE
        
        # 💡 [핵심 비선형 감쇄 적용]
        # 점수가 base_score를 기준으로 위아래로 움직이되, 100점(또는 0점) 근처로 갈수록 저항이 생겨 
        # 무한정 치솟지 않고 부드럽게 수렴(Saturation)하도록 탄젠트(tanh) 변환 활용
        # 공식: base + amplitude * tanh(linear_shift / TANH_SATURATION_DIVISOR)
        # base_score가 50이 아닌 값으로 설정되더라도 0~100 범위를 벗어나지 않도록
        # amplitude를 base와 100-base 중 작은 쪽으로 자동 제한
        base = betas.get('base_score', 50.0)
        amplitude = min(base, 100.0 - base)
        saturated_score = base + amplitude * np.tanh(linear_shift / TANH_SATURATION_DIVISOR)
        
        # 안전하게 SCORE_FLOOR ~ SCORE_CEIL 범위 클리핑
        final_score = max(SCORE_FLOOR, min(SCORE_CEIL, saturated_score))
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

    # 기상청 단기예보(격자 단위, ~2~3일치) 수집 -> nx,ny 조합별로 한 번씩만 호출
    print("\n📡 [API 연동] 기상청 단기예보(기온/습도) 격자별 수집 중...")
    unique_grids = {(loc['nx'], loc['ny']) for loc in CHEONAN_DONGS}
    short_wx_cache = {}
    for nx, ny in unique_grids:
        short_wx_cache[(nx, ny)] = fetch_kma_short_forecast(nx, ny, KMA_AUTH_KEY)

    # 기상청 중기예보(권역 단위, D+3~D+10 기온) 수집 -> 천안 전역에 공통 적용
    mid_wx_map = fetch_kma_mid_temp(KMA_MID_TA_REG_ID, KMA_AUTH_KEY)

    # 중기예보 구간(습도 미제공)에 대비해 nx,ny별 '마지막으로 확보한 실제 습도'를 기억해뒀다가 이어씀
    last_known_humidity = {}

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

            # ---------- 기온/습도/일교차: 단기예보(우선) -> 중기예보(습도는 마지막 실측 유지) -> 해시 시뮬레이션(최후 폴백) ----------
            grid_key = (loc['nx'], loc['ny'])
            short_day = short_wx_cache.get(grid_key, {}).get(date_str)

            if short_day and short_day.get('diurnal') is not None:
                simulated_temp = short_day['temp']
                simulated_hum = short_day['humidity']
                simulated_diurnal = short_day['diurnal']
                last_known_humidity[grid_key] = simulated_hum
                weather_source = '단기예보(실측기반)'
            else:
                mid_day = mid_wx_map.get(date_str)
                if mid_day:
                    simulated_temp = mid_day['temp']
                    simulated_diurnal = mid_day['diurnal']
                    # 중기예보는 습도를 제공하지 않음 -> 이 격자에서 가장 최근에 확보한 습도를 이어씀
                    # (한 번도 확보 못했다면 질병모델의 기준습도 75.0을 중립값으로 사용)
                    simulated_hum = last_known_humidity.get(grid_key, 75.0)
                    weather_source = '중기예보(습도는 직전값 유지)'
                else:
                    # 최후 폴백: 단기/중기 모두 실패했을 때만 결정론적 시뮬레이션 사용
                    simulated_temp = 25.5 + ((_stable_hash(date_str + dong) % 9) - 4)
                    simulated_hum = 75.0 + ((_stable_hash(dong + date_str) % 30) - 15)
                    simulated_diurnal = 8.5 + ((_stable_hash(date_str) % 7) - 3)
                    weather_source = '예보없음(시뮬레이션폴백)'

            current_disease_scores = calculate_disease_risks(
                simulated_temp, simulated_hum, daily_pm25, simulated_diurnal
            )
            
            # 시계열 스무딩: pm25_confidence에 따라 '오늘 값'을 얼마나 반영할지 다르게 적용
            # (기존엔 신뢰도와 무관하게 항상 0.4:0.6 고정 비율이라, 예보 등급 기반 추정치나
            #  예보 없음(지속성가정)까지 실측치와 동일한 비중으로 반영되는 문제가 있었음)
            current_weight = CONFIDENCE_CURRENT_WEIGHT.get(pm25_confidence, 0.6)
            loc_key = f"{district}_{dong}"
            smoothed_scores = {}
            if loc_key in previous_scores_memory:
                prev_scores = previous_scores_memory[loc_key]
                for d_name, cur_val in current_disease_scores.items():
                    prev_val = prev_scores.get(d_name, cur_val)
                    smoothed_val = ((1 - current_weight) * prev_val) + (current_weight * cur_val)
                    smoothed_scores[d_name] = round(float(smoothed_val), 1)
            else:
                smoothed_scores = current_disease_scores
                
            previous_scores_memory[loc_key] = smoothed_scores
            
            score_values = list(smoothed_scores.values())
            mean_risk = np.mean(score_values)
            max_risk = np.max(score_values)
            
            # Max-Driven 하이브리드 Total Risk
            total_risk = round((0.5 * mean_risk) + (0.5 * max_risk), 1)
            
            # Red Flag 경고 체계 (WARNING_THRESHOLD점 이상 치솟는 질환 감지)
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
                'pm25_confidence': pm25_confidence,
                'weather_source': weather_source
            })
            
            detail_row = {
                'date': date_str,
                'district': district,
                'dong': dong,
                'station': station,
                'pm25_actual': daily_pm25,
                'pm25_source': pm25_grade,
                'pm25_confidence': pm25_confidence,
                'weather_source': weather_source,
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