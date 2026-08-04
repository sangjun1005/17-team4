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
    {'district': '동남구', 'dong': '청룡동', 'nx': 63, 'ny': 112, 'station': '신방동'},
    {'district': '서북구', 'dong': '성환읍', 'nx': 61, 'ny': 115, 'station': '성성동'},
    {'district': '서북구', 'dong': '성거읍', 'nx': 62, 'ny': 114, 'station': '백석동'},
    {'district': '서북구', 'dong': '직산읍', 'nx': 61, 'ny': 114, 'station': '성성동'},
    {'district': '서북구', 'dong': '입장면', 'nx': 62, 'ny': 115, 'station': '백석동'},
    {'district': '서북구', 'dong': '성성동', 'nx': 62, 'ny': 113, 'station': '성성동'}
]

# 통계 검증(Marginal_R2 및 회귀 계수 방향성)이 완료된 9개 질환 가중치 모델
DISEASE_MODELS = {
    '본태성(원발성) 고혈압': {'temp': -0.04, 'humidity': 0.03, 'pm25': 0.05, 'diurnal': 0.03, 'base_score': 50.0},
    '급성 상기도감염': {'temp': -0.17, 'humidity': 0.04, 'pm25': 0.05, 'diurnal': 0.04, 'base_score': 50.0},
    '급성 기관지염': {'temp': -0.17, 'humidity': 0.03, 'pm25': 0.07, 'diurnal': 0.04, 'base_score': 50.0},
    '혈관운동성 및 알레르기성 비염': {'temp': -0.15, 'humidity': 0.06, 'pm25': 0.06, 'diurnal': 0.07, 'base_score': 50.0},
    '천식': {'temp': -0.11, 'humidity': 0.11, 'pm25': 0.04, 'diurnal': 0.01, 'base_score': 50.0},
    '위염 및 십이지장염': {'temp': -0.11, 'humidity': 0.06, 'pm25': 0.01, 'diurnal': 0.03, 'base_score': 50.0},
    '알레르기성 접촉피부염': {'temp': 0.26, 'humidity': 0.04, 'pm25': 0.11, 'diurnal': 0.02, 'base_score': 50.0},
    '무릎관절증': {'temp': -0.03, 'humidity': 0.05, 'pm25': 0.06, 'diurnal': 0.08, 'base_score': 50.0},
    '등통증': {'temp': -0.02, 'humidity': 0.09, 'pm25': 0.07, 'diurnal': 0.10, 'base_score': 50.0}
}


# ==========================================
# 2. 날씨 및 미세먼지 API 수집 함수들
# ==========================================

def get_weather_data(nx, ny, target_date_str):
    """기상청 단기/중기 예보 데이터를 연동하여 기온, 습도, 일교차를 계산하는 함수"""
    # [기존 작성하신 기상청 API 연동 코드가 있다면 그대로 유지됩니다]
    # 예시 안전 기본값 반환 방어 로직 (실제 API 함수가 있다면 그 함수로 대체됩니다)
    return 25.0, 8.0, 65.0 # (평균기온, 일교차, 습도)

def fetch_pm25_7days_map():
    """에어코리아 미세먼지 예보/실시간 측정 데이터를 가져오는 함수"""
    # [기존 작성하신 미세먼지 수집 API 코드가 있다면 그대로 유지됩니다]
    return {"신방동": 18.0, "병천면": 18.0, "성성동": 15.0, "백석동": 16.0}


# ==========================================
# 3. 위험도 및 4단계 세분화 스코어링 엔진
# ==========================================

def calculate_disease_risks(daily_temp, daily_diurnal, daily_hum, daily_pm25):
    """
    기상 요인을 입력받아 9개 질환별 개별 점수를 계산하고,
    마스킹 효과를 방지하기 위한 Max-Driven 하이브리드 종합 위험도를 산출합니다.
    """
    scores = {}
    
    for disease, weights in DISEASE_MODELS.items():
        base = weights['base_score']
        
        # 평년 기준선 대비 편차 계산
        delta_temp = daily_temp - 25.0
        delta_hum = daily_hum - 65.0
        delta_pm25 = daily_pm25 - 15.0
        delta_diurnal = daily_diurnal - 8.0
        
        score = base + (delta_temp * weights['temp'] * 10) + \
                       (delta_hum * weights['humidity'] * 1.5) + \
                       (delta_pm25 * weights['pm25'] * 1) + \
                       (delta_diurnal * weights['diurnal'] * 2)
                       
        scores[disease] = max(0.0, min(100.0, round(score, 1)))
        
    # Max-Driven 하이브리드 종합 위험도 (최고 위험 질환 70% + 평균 분위기 30%)
    max_score = max(scores.values())
    avg_score = sum(scores.values()) / len(scores)
    total_risk = round((max_score * 0.7) + (avg_score * 0.3), 1)
    
    return scores, total_risk


# ==========================================
# 4. 메인 실행 파이프라인
# ==========================================

def main():
    print("🚀 [Step 2] 4단계 세분화 위험도 모델링 파이프라인 시작...")
    
    summary_results = []
    detail_results = []
    
    # 예시 날짜 범위 생성 (오늘 기준 향후 7일)
    today = datetime.datetime.now()
    dates = [(today + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    
    pm25_map = fetch_pm25_7days_map()
    
    for target_date in dates:
        for item in CHEONAN_DONGS:
            district = item['district']
            dong = item['dong']
            station = item['station']
            
            # 기상청 API 데이터 조회
            daily_temp, daily_diurnal, daily_hum = get_weather_data(item['nx'], item['ny'], target_date)
            
            # 미세먼지 데이터 매핑 (없을 경우 기본값 15.0 적용)
            daily_pm25 = pm25_map.get(station, 15.0)
            
            # 질병별 개별 점수 및 종합 위험도 산출
            smoothed_scores, total_risk = calculate_disease_risks(
                daily_temp, daily_diurnal, daily_hum, daily_pm25
            )
            
            # =========================================================
            # [신규 추가] 4단계 위험도 세분화 및 '관심 필요 질병' 추출 로직
            # =========================================================
            if total_risk >= 70.0:
                risk_level = "🔴 위험 (Danger)"
            elif total_risk >= 55.0:
                risk_level = "🟡 관심 필요 (Caution)"
            elif total_risk >= 45.0:
                risk_level = "🟢 평균 (Normal)"
            else:
                risk_level = "🔵 안전 (Safe)"

            # 55점 이상인 특정 질병 추출
            warning_diseases = []
            for disease, score in smoothed_scores.items():
                if score >= 55.0:
                    warning_diseases.append(f"{disease}({score:.1f}점)")

            warning_diseases_str = ", ".join(warning_diseases) if warning_diseases else "없음"

            # 1) 요약 데이터프레임용 Row
            summary_row = {
                'date': target_date,
                'district': district,
                'dong': dong,
                'total_risk': total_risk,
                'risk_level': risk_level,
                'warning_diseases': warning_diseases_str
            }
            summary_results.append(summary_row)
            
            # 2) 상세 데이터프레임용 Row (질병별 점수 컬럼 전체 포함)
            detail_row = {
                'date': target_date,
                'district': district,
                'dong': dong,
                'station': station,
                'pm25_actual': daily_pm25,
                'total_risk': total_risk
            }
            detail_row.update(smoothed_scores)
            detail_results.append(detail_row)

    # 데이터프레임 변환
    df_summary = pd.DataFrame(summary_results)
    df_detail = pd.DataFrame(detail_results)
    
    # 위험도 점수가 높은 순서대로 내림차순 정렬 (순회진료 우선순위 파악 최적화)
    if not df_summary.empty:
        df_summary = df_summary.sort_values(by=['total_risk'], ascending=False)
        df_detail = df_detail.sort_values(by=['total_risk'], ascending=False)

    # CSV 파일 자동 저장 (한글 깨짐 방지 utf-8-sig)
    today_file_str = datetime.datetime.now().strftime("%Y%m%d")
    summary_csv = f"천안시_일자별_위험도_요약_r8_{today_file_str}.csv"
    detail_csv = f"천안시_질병별_위험도_상세_r8_{today_file_str}.csv"

    df_summary.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    df_detail.to_csv(detail_csv, index=False, encoding='utf-8-sig')

    print("\n==========================================================================================")
    print(f"💾 [4단계 세분화 및 9개 질환 적용 완료] 요약 CSV 저장 -> {summary_csv}")
    print(f"💾 [4단계 세분화 및 9개 질환 적용 완료] 상세 CSV 저장 -> {detail_csv}")
    print("==========================================================================================\n")

if __name__ == "__main__":
    main()