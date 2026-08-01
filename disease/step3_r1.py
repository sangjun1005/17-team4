import os
import pickle
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 0. 한글 폰트 및 마이너스 부호 설정
# ---------------------------------------------------------
sns.set_theme(style="whitegrid")
system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 1. 경로 설정 및 질병 액션 맵
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
models_dir = os.path.join(BASE_DIR, 'saved_models')

DISEASE_ACTION_MAP = {
    'M17': {'명칭': '무릎관절증', '지참물품': '온열 파스/찜질팩, 관절 통증완화제'},
    'J45': {'명칭': '천식', '지참물품': '호흡기 흡입제, 방진마스크, 폐기능 측정기'},
    'J20': {'명칭': '급성기관지염', '지참물품': '진해거담제, 체온계, 진료 청진기'},
    'I10': {'명칭': '고혈압', '지참물품': '자동 혈압계, 복용 지도서'},
    'J30': {'명칭': '알레르기 비염', '지참물품': '항히스타민제, 비강 분무액'},
    'E11': {'명칭': '당뇨병', '지참물품': '혈당측정기, 당뇨 소모품'},
    'J06': {'명칭': '급성상기도감염', '지참물품': '해열진통제, 감기약'},
    'K29': {'명칭': '위염및위십이지장염', '지참물품': '제산제, 위장약'}
}

# ---------------------------------------------------------
# 2. 10일 기상 예보 데이터 수집 (기상 변동폭 실감형 테스트)
# ---------------------------------------------------------
dates = pd.date_range(start='2026-08-01', periods=10, freq='D')
df_forecast = pd.DataFrame({'날짜': dates})

# 8월 평년 기상 대비 편차가 발생하는 시나리오 (3~4일차 폭염/미세먼지 급증)
df_forecast['월평균기온'] = [26.5, 27.0, 33.5, 35.5, 29.0, 26.0, 24.0, 26.0, 26.5, 27.0]
df_forecast['월최대_PM25'] = [15.0, 18.0, 55.0, 80.0, 25.0, 15.0, 10.0, 15.0, 16.0, 15.0]
df_forecast['월평균습도'] = [70.0, 72.0, 88.0, 94.0, 78.0, 70.0, 65.0, 70.0, 71.0, 70.0]

# ---------------------------------------------------------
# 3. 보건기상지수 방식의 동적 스코어링 엔진 (Dynamic Health Scaling)
# ---------------------------------------------------------
disease_scores = {}
disease_full_names = {}

# 기상 변수별 통계적 표준편차 (Z-Score 정규화용 표준 임계치)
STD_TEMP = 2.5   # 기온 변동 표준편차(℃)
STD_PM25 = 12.0  # 미세먼지 변동 표준편차(µg/m³)
STD_HUMID = 8.0  # 습도 변동 표준편차(%)

if os.path.exists(models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.startswith('model_') and f.endswith('.pkl')]
else:
    model_files = []

if model_files:
    for mfile in model_files:
        code = mfile.replace('model_', '').replace('.pkl', '')
        with open(os.path.join(models_dir, mfile), 'rb') as f:
            model_data = pickle.load(f)
            
        params = model_data.get('params', {})
        baseline = model_data.get('monthly_baseline', {})
        d_name = DISEASE_ACTION_MAP.get(code, {}).get('명칭', model_data.get('disease_name', code))
        
        full_label = f"{d_name} ({code})"
        disease_full_names[code] = full_label
        
        base_temp = baseline.get(8, {}).get('월평균기온', 26.5)
        base_pm25 = baseline.get(8, {}).get('월최대_PM25', 18.0)
        base_humid = baseline.get(8, {}).get('월평균습도', 70.0)
        
        # [핵심 1] Z-score 기반 정규화 편차 산출
        z_temp = (df_forecast['월평균기온'] - base_temp) / STD_TEMP
        z_pm25 = (df_forecast['월최대_PM25'] - base_pm25) / STD_PM25
        z_humid = (df_forecast['월평균습도'] - base_humid) / STD_HUMID
        
        b_temp = params.get('월평균기온_lag1_scaled', 0)
        b_pm25 = params.get('월최대_PM25_lag1_scaled', 0)
        b_humid = params.get('월평균습도_lag1_scaled', 0)
        
        # 회귀 계수와 Z-score 결합
        raw_risk_delta = (z_temp * b_temp) + (z_pm25 * b_pm25) + (z_humid * b_humid)
        
        # [핵심 2] 동적 민감도 민감성 증폭 (Scaling Factor 40.0 적용)
        # 평년 수준(0)일 땐 50점, 날씨 악화 시 80~90점대, 쾌적할 땐 20~30점대로 넓게 확산
        dynamic_score = 50.0 + (raw_risk_delta * 40.0)
        disease_scores[code] = np.clip(np.round(dynamic_score, 1), 0.0, 100.0)
else:
    # 예시용 테스트 세팅
    for code, info in DISEASE_ACTION_MAP.items():
        full_label = f"{info['명칭']} ({code})"
        disease_full_names[code] = full_label
        if code in ['M17', 'J45']: # 기상 민감 질환
            disease_scores[code] = [50.0, 54.0, 84.5, 95.0, 65.0, 50.0, 32.0, 50.0, 51.0, 50.0]
        else:
            disease_scores[code] = [50.0, 52.0, 71.0, 82.0, 58.0, 50.0, 38.0, 50.0, 50.5, 50.0]

# ---------------------------------------------------------
# 4. 리포트 데이터프레임 구축
# ---------------------------------------------------------
df_report = pd.DataFrame({'날짜': df_forecast['날짜'].dt.strftime('%Y-%m-%d')})

score_cols = []
for code, full_label in disease_full_names.items():
    df_report[full_label] = disease_scores[code]
    score_cols.append(full_label)

# [핵심 3] 기상청/CAI 스타일 종합 점수 (Max 60% + Mean 40% 혼합)
max_scores = df_report[score_cols].max(axis=1)
mean_scores = df_report[score_cols].mean(axis=1)
df_report['종합_추천점수'] = np.round((max_scores * 0.6) + (mean_scores * 0.4), 1)

# 위험 질병군 및 준비 물품 판별 (보건기상지수 4단계 구조 도입)
high_risk_list = []
action_items_list = []

for idx, row in df_report.iterrows():
    danger_diseases = []
    actions = set()
    
    for code, full_label in disease_full_names.items():
        score = row[full_label]
        d_name = DISEASE_ACTION_MAP.get(code, {}).get('명칭', code)
        
        if score >= 80:
            danger_diseases.append(f"[매우높음/고위험]{d_name}({score}점)")
            if code in DISEASE_ACTION_MAP:
                actions.add(DISEASE_ACTION_MAP[code]['지참물품'])
        elif score >= 65:
            danger_diseases.append(f"[높음/주의]{d_name}({score}점)")
            if code in DISEASE_ACTION_MAP:
                actions.add(DISEASE_ACTION_MAP[code]['지참물품'])
                
    if danger_diseases:
        high_risk_list.append(", ".join(danger_diseases))
        action_items_list.append(" / ".join(actions))
    else:
        high_risk_list.append("특이 위험 질환 없음 (보통/낮음 단계)")
        action_items_list.append("기본 건강검진 장비 지참")

df_report['위험_질병군_목록'] = high_risk_list
df_report['순회진료_추천자재'] = action_items_list

# ---------------------------------------------------------
# 5. 콘솔 리포트 출력
# ---------------------------------------------------------
print("=" * 120)
print(" 📊 보건기상지수 방식 적용 10일 순회진료 동적 추천 리포트")
print("=" * 120)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df_report[['날짜', '종합_추천점수'] + score_cols[:3] + ['위험_질병군_목록']])
print("=" * 120)

# ---------------------------------------------------------
# 6. 동적 시각화 차트 생성
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(20, 13))
fig.suptitle('🏥 보건기상지수 기반 10일 순회진료 동적 위험도 대시보드', fontsize=20, fontweight='bold', y=0.98)

# Chart 1: 질병별 10일 위험도 점수 추이 (변동성 확장)
ax1 = axes[0, 0]
for full_label in score_cols:
    ax1.plot(df_report['날짜'], df_report[full_label], marker='o', linewidth=2, label=full_label)

ax1.axhline(35, color='green', linestyle='--', alpha=0.6, label='낮음 (35점 이하)')
ax1.axhline(50, color='gray', linestyle='--', alpha=0.7, label='보통 (50점 기준)')
ax1.axhline(65, color='orange', linestyle='--', alpha=0.8, label='높음/주의 (65점)')
ax1.axhline(80, color='red', linestyle='--', alpha=0.8, label='매우높음/고위험 (80점)')

ax1.set_title('① 질병별 동적 위험도 점수 추이 (20~95점 확산)', fontsize=15, fontweight='bold')
ax1.set_ylabel('보건기상 위험도 점수', fontsize=12)
ax1.set_ylim(10, 100)
ax1.tick_params(axis='x', rotation=30)
ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1.02), fontsize=9)

# Chart 2: 날짜 x 질병 위험도 히트맵
ax2 = axes[0, 1]
heatmap_data = df_report.set_index('날짜')[score_cols].T
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd", cbar=True, ax=ax2, vmin=30, vmax=95, linewidths=0.5)
ax2.set_title('② 날짜 × 질병 위험도 히트맵 (명확한 핫스팟 구분)', fontsize=15, fontweight='bold')
ax2.tick_params(axis='y', rotation=0)
ax2.tick_params(axis='x', rotation=30)

# Chart 3: 날짜별 종합 추천점수 (Max-Mean 적용)
ax3 = axes[1, 0]
colors = ['#d9534f' if s >= 80 else ('#f0ad4e' if s >= 65 else '#5bc0de') for s in df_report['종합_추천점수']]
bars = ax3.bar(df_report['날짜'], df_report['종합_추천점수'], color=colors, width=0.5)

ax3.axhline(50, color='gray', linestyle='--', label='평년 기준 (50점)')
ax3.set_title('③ 일자별 종합 추천점수 (최고 위험도 우선 가중)', fontsize=15, fontweight='bold')
ax3.set_ylabel('종합 추천점수', fontsize=12)
ax3.set_ylim(0, 100)
ax3.tick_params(axis='x', rotation=30)

for bar in bars:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Chart 4: 기상 변수 Z-Score 편차 추이
ax4 = axes[1, 1]
base_temp = 26.5
base_pm25 = 18.0

z_temp_plot = (df_forecast['월평균기온'] - base_temp) / STD_TEMP
z_pm25_plot = (df_forecast['월최대_PM25'] - base_pm25) / STD_PM25

ax4.plot(df_report['날짜'], z_temp_plot, marker='s', color='red', label='기온 Z-Score 편차')
ax4.plot(df_report['날짜'], z_pm25_plot, marker='^', color='purple', label='미세먼지 Z-Score 편차')
ax4.axhline(0, color='black', linestyle='-', alpha=0.5)

ax4.set_title('④ 예보 기상 Z-Score 편차 (기상청 방식 정규화)', fontsize=15, fontweight='bold')
ax4.set_ylabel('표준편차 단위 변동 (Z-Score)', fontsize=12)
ax4.tick_params(axis='x', rotation=30)
ax4.legend(loc='upper right', fontsize=10)

plt.tight_layout()
output_img_path = os.path.join(BASE_DIR, '보건기상지수_동적_시각화_리포트.png')
plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
print(f"\n🖼️ 보건기상지수 방식 시각화 리포트가 저장되었습니다: {output_img_path}")
plt.show()