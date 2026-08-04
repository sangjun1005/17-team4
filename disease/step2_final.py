import re
import time
import hashlib
import datetime
import requests
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

# ==========================================
# 0. 설정 / 튜닝 상수
# ==========================================

AIR_SERVICE_KEY = "6069cd5378ffe531429bdca0ff28ff0f0b0e661007504bc6270cc80995a053b9"  # 공공데이터포털(에어코리아)
KMA_AUTH_KEY = "8wUF4lmCQuGFBeJZggLhhQ"  # 기상청 API허브(apihub.kma.go.kr) 전용 키 (공공데이터포털 키와 체계가 다름)

CHEONAN_STATIONS = ['성성동', '수신면', '신방동', '성황동', '성거읍', '백석동']

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
    {'district': '서북구', 'dong': '성성동', 'nx': 60, 'ny': 112, 'station': '성성동'},
]

# 질병별 회귀계수(beta): temp/humidity/pm25/diurnal, base_score. 괄호 안은 Marginal R2.
DISEASE_MODELS = {
    '본태성(원발성) 고혈압':          {'temp': -0.04, 'humidity': 0.03, 'pm25': 0.05, 'diurnal': 0.03, 'base_score': 50.0},  # R2 0.29
    '급성 상기도감염':               {'temp': -0.17, 'humidity': 0.04, 'pm25': 0.05, 'diurnal': 0.04, 'base_score': 50.0},  # R2 0.62
    '급성 기관지염':                 {'temp': -0.17, 'humidity': 0.03, 'pm25': 0.07, 'diurnal': 0.04, 'base_score': 50.0},  # R2 0.60
    '혈관운동성 및 알레르기성 비염':  {'temp': -0.15, 'humidity': 0.06, 'pm25': 0.06, 'diurnal': 0.07, 'base_score': 50.0},  # R2 0.68
    '천식':                         {'temp': -0.11, 'humidity': 0.11, 'pm25': 0.04, 'diurnal': 0.01, 'base_score': 50.0},  # R2 0.55
    '위염 및 십이지장염':            {'temp': -0.11, 'humidity': 0.06, 'pm25': 0.01, 'diurnal': 0.03, 'base_score': 50.0},  # R2 0.44
    '알레르기성 접촉피부염':         {'temp': 0.26,  'humidity': 0.04, 'pm25': 0.11, 'diurnal': 0.02, 'base_score': 50.0},  # R2 0.86
    '무릎관절증':                    {'temp': -0.03, 'humidity': 0.05, 'pm25': 0.06, 'diurnal': 0.08, 'base_score': 50.0},  # R2 0.43
    '등통증':                       {'temp': -0.02, 'humidity': 0.09, 'pm25': 0.07, 'diurnal': 0.10, 'base_score': 50.0},  # R2 0.56
}

# 위험도 점수화 튜닝 상수 (실제 운영 데이터로 추후 보정 권장)
LINEAR_SHIFT_SCALE = 20.0
TANH_SATURATION_DIVISOR = 35.0
SCORE_FLOOR, SCORE_CEIL = 10.0, 100.0
WARNING_THRESHOLD = 75.0

# 시계열 스무딩 시 '오늘 값'에 부여할 가중치 (신뢰도가 낮을수록 과거값에 더 의존)
CONFIDENCE_CURRENT_WEIGHT = {'high': 0.60, 'medium': 0.45, 'low': 0.30}

# 기상청 발표 주기 / 중기예보 지역코드
KMA_SHORT_ISSUE_TIMES = ['0200', '0500', '0800', '1100', '1400', '1700', '2000', '2300']
KMA_MID_ISSUE_TIMES = ['0600', '1800']
KMA_MID_TA_REG_ID = "11C20401"  # 대전(대전·세종·충남권 대표) — ⚠️ 천안 전용 코드인지 검증 필요

BASE_WX = {'temp': 25.5, 'humidity': 75.0, 'pm25': 18.0, 'diurnal': 8.5}  # 위험도 계산 기준값


def _stable_hash(text: str) -> int:
    """파이썬 내장 hash()는 실행마다 시드가 달라져 재현이 안 되므로, 동일 입력=동일 출력을 보장하는 해시로 대체."""
    return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)


# ==========================================
# 1. 공통 API 요청 헬퍼
# ==========================================

def _api_request(url, params, label, timeout=10, debug=False):
    """
    공공데이터포털 / 기상청 API허브 공통 XML 응답 처리.
    표준 에러 포맷(cmmMsgHeader) 및 resultCode 오류를 모두 확인.
    성공 시 item Element 리스트, 실패 시 None 반환.
    """
    try:
        response = requests.get(url, params=params, timeout=timeout)
        if response.status_code != 200:
            print(f"  └ [{label}] HTTP {response.status_code}")
            return None

        root = ET.fromstring(response.content)
        if debug:
            print(f"     [원문] {response.text[:1200]}")

        err = root.find('.//cmmMsgHeader')
        if err is not None:
            msg = err.findtext('errMsg', '') or err.findtext('returnAuthMsg', '')
            print(f"  └ [{label}] 에러: [{err.findtext('returnReasonCode', '')}] {msg}")
            return None

        rc = root.find('.//resultCode')
        if rc is not None and rc.text != '00':
            print(f"  └ [{label}] 오류코드 {rc.text} ({root.findtext('.//resultMsg', '')})")
            return None

        return root.findall('.//item')

    except ET.ParseError as e:
        print(f"  └ [{label}] XML 파싱 실패: {e}")
        return None
    except Exception as e:
        print(f"  └ [{label}] 요청 예외: {e}")
        return None


def _get_latest_kma_issue(issue_times, buffer_minutes=10):
    """현재 시각 기준으로 이미 발표됐을 가장 최근 기상청 발표시각(base_date, base_time)을 계산."""
    now = datetime.datetime.now() - datetime.timedelta(minutes=buffer_minutes)
    for t in sorted(issue_times, reverse=True):
        if (now.hour, now.minute) >= (int(t[:2]), int(t[2:])):
            return now.strftime("%Y%m%d"), t
    return (now - datetime.timedelta(days=1)).strftime("%Y%m%d"), sorted(issue_times)[-1]


# ==========================================
# 2. 미세먼지 수집 (실시간 + 주간예보 + 블렌딩)
# ==========================================

def fetch_cheonan_realtime_pm(service_key, stations=CHEONAN_STATIONS, debug=False):
    """측정소별 당일 실시간 PM2.5 수집. 실패 시 기본값 18.0."""
    url = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
    print("📡 실시간 미세먼지 수집 중...")
    result = {}
    for station in stations:
        params = {'serviceKey': service_key, 'returnType': 'xml', 'numOfRows': '10', 'pageNo': '1',
                   'stationName': station, 'dataTerm': 'DAILY', 'ver': '1.0'}
        items = _api_request(url, params, f"실시간:{station}", debug=debug)
        val = 18.0
        if items:
            try:
                val = float(items[0].findtext('pm25Value', ''))
            except (ValueError, TypeError):
                pass
        result[station] = val
        print(f"  └ [{station}] PM2.5 = {val}")
    return result


_WEEKLY_DAY_KEYS = [('frcstOneDt', 'frcstOneCn'), ('frcstTwoDt', 'frcstTwoCn'),
                     ('frcstThreeDt', 'frcstThreeCn'), ('frcstFourDt', 'frcstFourCn')]


def fetch_weekly_pm25_forecast(service_key, region_name='충남', debug=False):
    """
    초미세먼지 주간예보(낮음/높음)를 날짜별로 수집.
    1단계: searchDate 없이 호출 -> 실제 발표일자(presnatnDt) 목록 확보
    2단계: 최신 발표일자부터 순서대로 상세 조회 (최대 3개 날짜 x 2회 재시도, 504 등 일시오류 대비)
    """
    url = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMinuDustWeekFrcstDspth"
    region_pattern = re.compile(rf'{region_name}\s*:\s*([^\s,]+)')
    base_params = {'serviceKey': service_key, 'returnType': 'xml', 'numOfRows': '100', 'pageNo': '1'}
    grade_map = {}

    items = _api_request(url, base_params, "주간예보:발표일목록", debug=debug)
    presnatn_dates = sorted({it.findtext('presnatnDt', '') for it in (items or []) if it.findtext('presnatnDt', '')},
                             reverse=True)
    if not presnatn_dates:
        today = datetime.datetime.now()
        presnatn_dates = [today.strftime("%Y-%m-%d"), (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d")]

    for search_date in presnatn_dates[:3]:
        params = {**base_params, 'searchDate': search_date}
        for retry in range(2):
            label = f"주간예보:{search_date}" + (f"(재시도{retry})" if retry else "")
            items = _api_request(url, params, label, debug=debug)
            if items is None:
                time.sleep(1)
                continue
            for item in items or []:
                for dt_tag, cn_tag in _WEEKLY_DAY_KEYS:
                    fdate, fcn = item.findtext(dt_tag, ''), item.findtext(cn_tag, '')
                    if not fdate or not fcn:
                        continue
                    m = region_pattern.search(fcn)
                    if not m:
                        continue
                    grade = m.group(1).rstrip('.,')
                    norm = fdate.replace('.', '-').replace('/', '-')
                    if re.fullmatch(r'\d{8}', norm):
                        norm = f"{norm[:4]}-{norm[4:6]}-{norm[6:8]}"
                    if grade in ('낮음', '높음'):
                        grade_map[norm] = grade
            break  # 응답을 받았으면(빈 결과라도) 이 날짜에 대한 재시도는 종료
        if grade_map:
            break

    print(f"  └ {region_name} 주간예보 확보: {grade_map}" if grade_map else f"  └ {region_name} 주간예보 실패 -> 지속성 가정")
    return grade_map


def estimate_future_pm25(today_actual, region_anchor, grade, day_offset,
                          decay_rate=0.6, low_anchor=20.0, high_anchor=45.0):
    """
    오늘 실측 PM2.5(측정소, 고해상도) + 충남 권역 예보등급(낮음/높음, 저해상도)을 결합.
    측정소가 지역평균 대비 갖는 편차(offset)를 day_offset에 따라 감쇠시키며 등급별 대표값에 수렴.
    """
    offset = (today_actual - region_anchor) * (decay_rate ** day_offset)
    if grade == '낮음':
        anchor, bounds = low_anchor, (0.0, 35.0)
    elif grade == '높음':
        anchor, bounds = high_anchor, (36.0, 150.0)
    else:
        anchor, bounds = 18.0, (0.0, 150.0)  # 예보 없음: 기준값으로 서서히 회귀
    return round(float(max(bounds[0], min(bounds[1], anchor + offset))), 1)


# ==========================================
# 3. 기상 수집 (단기예보 + 중기예보)
# ==========================================

def fetch_kma_short_forecast(nx, ny, auth_key, debug=False):
    """단기예보(격자단위, 보통 최대 2~3일치)로 날짜별 평균기온/평균습도/일교차 수집."""
    url = "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst"
    base_date, base_time = _get_latest_kma_issue(KMA_SHORT_ISSUE_TIMES)
    params = {'authKey': auth_key, 'pageNo': '1', 'numOfRows': '1000', 'dataType': 'XML',
              'base_date': base_date, 'base_time': base_time, 'nx': str(nx), 'ny': str(ny)}
    items = _api_request(url, params, f"단기예보({nx},{ny})", debug=debug)
    if not items:
        return {}

    daily = {}
    for item in items:
        cat, fdate, fval = item.findtext('category', ''), item.findtext('fcstDate', ''), item.findtext('fcstValue', '')
        if not fdate or fval == '':
            continue
        bucket = daily.setdefault(f"{fdate[:4]}-{fdate[4:6]}-{fdate[6:8]}", {'TMP': [], 'REH': [], 'TMN': None, 'TMX': None})
        try:
            val = float(fval)
        except ValueError:
            continue
        if cat in ('TMP', 'REH'):
            bucket[cat].append(val)
        elif cat in ('TMN', 'TMX'):
            bucket[cat] = val

    result = {}
    for date_str, b in daily.items():
        if not b['TMP'] or not b['REH']:
            continue
        if b['TMN'] is not None and b['TMX'] is not None:
            diurnal = b['TMX'] - b['TMN']
        elif len(b['TMP']) >= 2:
            diurnal = max(b['TMP']) - min(b['TMP'])
        else:
            diurnal = None
        result[date_str] = {
            'temp': round(float(np.mean(b['TMP'])), 1),
            'humidity': round(float(np.mean(b['REH'])), 1),
            'diurnal': round(diurnal, 1) if diurnal is not None else None,
        }
    print(f"  └ 단기예보({nx},{ny}) {len(result)}일치 확보")
    return result


def fetch_kma_mid_temp(reg_id, auth_key, debug=False):
    """중기예보 기온(getMidTa)로 D+3~D+10 최고/최저기온 수집 (습도는 미제공)."""
    url = "https://apihub.kma.go.kr/api/typ02/openApi/MidFcstInfoService/getMidTa"
    base_date, base_time = _get_latest_kma_issue(KMA_MID_ISSUE_TIMES)
    params = {'authKey': auth_key, 'pageNo': '1', 'numOfRows': '10', 'dataType': 'XML',
              'regId': reg_id, 'tmFc': f"{base_date}{base_time}"}
    items = _api_request(url, params, f"중기예보({reg_id})", debug=debug)
    if not items:
        return {}

    item = items[0]
    issue_date = datetime.datetime.strptime(base_date, "%Y%m%d")
    result = {}
    for n in range(3, 11):
        tmin_text, tmax_text = item.findtext(f'taMin{n}', ''), item.findtext(f'taMax{n}', '')
        if tmin_text == '' or tmax_text == '':
            continue
        try:
            tmin_v, tmax_v = float(tmin_text), float(tmax_text)
        except ValueError:
            continue
        target = (issue_date + datetime.timedelta(days=n)).strftime("%Y-%m-%d")
        result[target] = {'temp': round((tmin_v + tmax_v) / 2, 1), 'diurnal': round(tmax_v - tmin_v, 1)}
    print(f"  └ 중기예보({reg_id}) {len(result)}일치 확보")
    return result


def resolve_daily_weather(date_str, grid_key, dong, short_cache, mid_map, last_known_humidity):
    """단기예보 -> 중기예보(습도는 직전값 유지) -> 결정론적 시뮬레이션(최후 폴백) 순으로 기상값 결정."""
    day = short_cache.get(grid_key, {}).get(date_str)
    if day and day['diurnal'] is not None:
        last_known_humidity[grid_key] = day['humidity']
        return day['temp'], day['humidity'], day['diurnal'], '단기예보(실측기반)'

    mid = mid_map.get(date_str)
    if mid:
        hum = last_known_humidity.get(grid_key, BASE_WX['humidity'])
        return mid['temp'], hum, mid['diurnal'], '중기예보(습도는 직전값 유지)'

    temp = 25.5 + ((_stable_hash(date_str + dong) % 9) - 4)
    hum = 75.0 + ((_stable_hash(dong + date_str) % 30) - 15)
    diurnal = 8.5 + ((_stable_hash(date_str) % 7) - 3)
    return temp, hum, diurnal, '예보없음(시뮬레이션폴백)'


# ==========================================
# 4. 질병 위험도 스코어링
# ==========================================

def calculate_disease_risks(obs_temp, obs_hum, obs_pm25, obs_diurnal):
    """
    질병별 회귀계수(beta) 기반 선형결합 + tanh 포화(saturation)로 0~100점 위험도 산출.
    base_score를 중심으로 대칭 진폭(min(base, 100-base))을 적용해 항상 0~100 범위 보장.
    """
    deltas = {
        'temp': obs_temp - BASE_WX['temp'],
        'humidity': obs_hum - BASE_WX['humidity'],
        'pm25': obs_pm25 - BASE_WX['pm25'],
        'diurnal': obs_diurnal - BASE_WX['diurnal'],
    }
    scores = {}
    for disease, betas in DISEASE_MODELS.items():
        impact = sum(deltas[k] * betas[k] for k in ('temp', 'humidity', 'pm25', 'diurnal'))
        linear_shift = impact * LINEAR_SHIFT_SCALE
        base = betas['base_score']
        amplitude = min(base, 100.0 - base)
        score = base + amplitude * np.tanh(linear_shift / TANH_SATURATION_DIVISOR)
        scores[disease] = round(float(max(SCORE_FLOOR, min(SCORE_CEIL, score))), 1)
    return scores


def resolve_daily_pm25(day_offset, date_str, station, realtime_pm_map, region_anchor, grade_map):
    """당일은 실측치 그대로, 이후는 주간예보 등급 기반 추정치 + 신뢰도 라벨 반환."""
    station_actual = realtime_pm_map.get(station, 18.0)
    if day_offset == 0:
        return station_actual, '실시간', 'high'
    grade = grade_map.get(date_str)
    pm25 = estimate_future_pm25(station_actual, region_anchor, grade, day_offset)
    return pm25, (grade or '예보없음(지속성가정)'), ('medium' if grade else 'low')


def build_grade_persistence_map(dates, weekly_grade_map):
    """예보 없는 날짜는 직전에 확인된 등급을 지속(persistence) 적용."""
    result, last_grade = {}, None
    for i, date_str in enumerate(dates):
        if i == 0:
            continue
        grade = weekly_grade_map.get(date_str)
        last_grade = grade if grade else last_grade
        result[date_str] = grade or last_grade
    return result


# ==========================================
# 5. 메인 파이프라인
# ==========================================

def main():
    print("🚀 [Step2_r9] 천안시 질병 위험도 예측 파이프라인 실행 중...")

    realtime_pm_map = fetch_cheonan_realtime_pm(AIR_SERVICE_KEY)
    weekly_grade_map = fetch_weekly_pm25_forecast(AIR_SERVICE_KEY, region_name='충남')
    region_anchor = float(np.mean(list(realtime_pm_map.values()))) if realtime_pm_map else 18.0

    print("📡 기상청 단기예보(격자별 기온/습도) 수집 중...")
    unique_grids = {(loc['nx'], loc['ny']) for loc in CHEONAN_DONGS}
    short_wx_cache = {grid: fetch_kma_short_forecast(*grid, KMA_AUTH_KEY) for grid in unique_grids}
    mid_wx_map = fetch_kma_mid_temp(KMA_MID_TA_REG_ID, KMA_AUTH_KEY)
    last_known_humidity = {}

    dates = [(datetime.datetime.now() + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
    grade_persistence_map = build_grade_persistence_map(dates, weekly_grade_map)

    previous_scores = {}
    summary_rows, detail_rows = [], []

    for day_offset, date_str in enumerate(dates):
        for loc in CHEONAN_DONGS:
            district, dong, station = loc['district'], loc['dong'], loc['station']

            pm25, pm25_source, pm25_conf = resolve_daily_pm25(
                day_offset, date_str, station, realtime_pm_map, region_anchor, grade_persistence_map)
            temp, hum, diurnal, wx_source = resolve_daily_weather(
                date_str, (loc['nx'], loc['ny']), dong, short_wx_cache, mid_wx_map, last_known_humidity)

            scores = calculate_disease_risks(temp, hum, pm25, diurnal)

            # 신뢰도 기반 시계열 스무딩: 신뢰도 낮은 추정치일수록 과거값에 더 의존
            loc_key = f"{district}_{dong}"
            current_weight = CONFIDENCE_CURRENT_WEIGHT.get(pm25_conf, 0.6)
            prev = previous_scores.get(loc_key)
            if prev:
                scores = {k: round((1 - current_weight) * prev.get(k, v) + current_weight * v, 1)
                          for k, v in scores.items()}
            previous_scores[loc_key] = scores

            vals = list(scores.values())
            mean_risk, max_risk = float(np.mean(vals)), float(np.max(vals))
            total_risk = round(0.5 * mean_risk + 0.5 * max_risk, 1)
            high_risk = [f"{k}({v}점)" for k, v in scores.items() if v >= WARNING_THRESHOLD]

            row = dict(date=date_str, district=district, dong=dong, station=station,
                       pm25_actual=pm25, pm25_source=pm25_source, pm25_confidence=pm25_conf,
                       weather_source=wx_source, total_risk=total_risk)

            summary_rows.append({**row, 'mean_risk': round(mean_risk, 1), 'max_risk': round(max_risk, 1),
                                  'is_warning': bool(high_risk),
                                  'high_risk_diseases': ", ".join(high_risk) or "안전",
                                  'temp': temp, 'humidity': hum})
            detail_rows.append({**row, **scores})

    df_summary = pd.DataFrame(summary_rows).sort_values(by=['is_warning', 'total_risk'], ascending=[False, False])
    df_detail = pd.DataFrame(detail_rows)

    today_str = datetime.datetime.now().strftime("%Y%m%d")
    summary_csv = f"천안시_일자별_위험도_요약_r9_{today_str}.csv"
    detail_csv = f"천안시_질병별_위험도_상세_r9_{today_str}.csv"
    df_summary.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    df_detail.to_csv(detail_csv, index=False, encoding='utf-8-sig')

    print(f"\n💾 요약 CSV -> {summary_csv}")
    print(f"💾 상세 CSV -> {detail_csv}")
    print("\n📊 [순회진료 추천 상위 5개 지역 및 일자]")
    print(df_summary[['date', 'district', 'dong', 'total_risk', 'high_risk_diseases']].head(5).to_string(index=False))


if __name__ == "__main__":
    main()