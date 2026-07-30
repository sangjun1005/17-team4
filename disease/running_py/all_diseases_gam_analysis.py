import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pygam import PoissonGAM, s, l

warnings.filterwarnings('ignore')

# -------------------------------------------------------------
# 0. 한글 폰트 설정 (Windows / Mac)
# -------------------------------------------------------------
if os.name == 'nt':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='AppleGothic')
plt.rc('axes', unicode_minus=False)

# -------------------------------------------------------------
# 1. 파일 경로 설정 및 데이터 로드
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, 'disease_weather_air_merged.csv')

if not os.path.exists(file_path):
    print(f"❌ [오류] '{file_path}' 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")
    exit()

df = pd.read_csv(file_path)
print(f"📌 [1/4] 원천 데이터 로드 완료 (총 {len(df)}행)")

# 결과 저장용 폴더 생성
output_dir = os.path.join(BASE_DIR, 'analysis_results')
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------------
# 2. 전처리 파이프라인 (코로나19 보정 + 트렌드/Lag 변수)
# -------------------------------------------------------------
df['진료년월_dt'] = pd.to_datetime(df['진료년월'].astype(str), format='%Y%m', errors='coerce')
if df['진료년월_dt'].isna().sum() > 0:
    df['진료년월_dt'] = pd.to_datetime(df['진료년월'].astype(str), errors='coerce')

df = df.sort_values(by=['질병코드', '진료년월_dt']).reset_index(drop=True)

df['연도'] = df['진료년월_dt'].dt.year
df['월'] = df['진료년월_dt'].dt.month

# Lag 변수 생성
df['환자수_lag12'] = df.groupby('질병코드')['환자수'].shift(12)

weather_cols = [
    '월평균기온', '월평균최저기온', '월평균최고기온', '월평균일교차', 
    '월평균습도', '월평균기압', '월평균_PM10', '월최대_PM10', 
    '월평균_PM25', '월최대_PM25'
]

# 실제로 데이터프레임에 존재하는 기상 변수만 추출
available_weather = [c for c in weather_cols if c in df.columns]

for col in available_weather:
    df[f'{col}_lag1'] = df.groupby('질병코드')[col].shift(1)

# 코로나19 기간(2020~2022) 제거
covid_mask = (df['진료년월_dt'] >= '2020-01-01') & (df['진료년월_dt'] <= '2022-12-31')
df_clean = df[~covid_mask].dropna().reset_index(drop=True)

print(f"📌 [2/4] 코로나19 보정 및 Lag 변수 전처리 완료 (남은 데이터: {len(df_clean)}행)")

# -------------------------------------------------------------
# 3. 데이터 내 모든 질병 목록 추출
# -------------------------------------------------------------
disease_list = df_clean['질병코드'].unique()
print(f"📌 [3/4] 분석 대상 질병 총 {len(disease_list)}개 발견: {list(disease_list)}\n")

summary_results = []

# -------------------------------------------------------------
# 4. 질병별 순회 분석 및 고도화 시각화 대시보드 저장
# -------------------------------------------------------------
print("==================================================================")
print("🚀 전체 질병 GAM 분석 및 비전공자용 종합 리포트 생성 시작")
print("==================================================================")

for idx, code in enumerate(disease_list, 1):
    sub_df = df_clean[df_clean['질병코드'] == code].copy().reset_index(drop=True)
    disease_name = sub_df['질병명'].iloc[0] if '질병명' in sub_df.columns and not sub_df.empty else code
    
    if len(sub_df) < 24: # 최소 2년 이상 데이터 필요
        print(f"⚠️ [{idx}/{len(disease_list)}] {code} ({disease_name}): 데이터 부족으로 스킵 ({len(sub_df)}행)")
        continue

    # GAM 변수 세팅
    X_cols = ['환자수_lag12', '연도'] + available_weather[:3] # 주요 기상 3개
    X = sub_df[X_cols].values
    y = sub_df['환자수'].values
    
    # pyGAM 학습
    try:
        gam = PoissonGAM(
            s(0, n_splines=5) + 
            l(1) + 
            s(2, n_splines=5) + 
            s(3, n_splines=5) + 
            s(4, n_splines=5)
        ).fit(X, y)
        
        r2 = gam.statistics_['pseudo_r2']['explained_deviance']
    except Exception as e:
        r2 = np.nan

    print(f"  [{idx}/{len(disease_list)}] {code} - {disease_name:15s} | GAM 설명력(R²): {r2*100:5.1f}%")

    # 결과 요약 기록
    summary_results.append({
        '질병코드': code,
        '질병명': disease_name,
        '데이터수': len(sub_df),
        'GAM_Pseudo_R2': round(r2, 4) if not np.isnan(r2) else None
    })

    # =========================================================
    # [비전공자 맞춤형 종합 시각화 대시보드 3D/3-Subplot]
    # =========================================================
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(f"[{code}] {disease_name} - 기상 요인 및 월별 연관성 종합 분석 대시보드", fontsize=16, fontweight='bold', y=0.98)

    # ---------------------------------------------------------
    # [차트 1: 왼쪽] 월별 기상변수-환자수 상관관계 히트맵 (진함 = 영향도 높음)
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    
    # 월별/기상요인별 상관계수 계산
    heatmap_cols = available_weather[:6]
    monthly_corr = sub_df.groupby('월')[heatmap_cols + ['환자수']].apply(
        lambda g: g[heatmap_cols].corrwith(g['환자수'])
    )
    
    sns.heatmap(monthly_corr.T, cmap='coolwarm', annot=True, fmt='.2f', 
                linewidths=0.5, ax=ax1, cbar_kws={'label': '상관계수 (진할수록 영향력 큼)'})
    ax1.set_title("① [비전공자용] 월별 기상요인 연관도 진하기 (Heatmap)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("월 (Month)")
    ax1.set_ylabel("기상 요인")

    # ---------------------------------------------------------
    # [차트 2: 오른쪽 위] 가장 영향력 높은 기상변수 vs 환자수 산포도 & 추세선
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    
    # 상관계수 절대값이 가장 높은 변수 선정
    main_weather = monthly_corr.abs().mean().idxmax()
    
    sns.scatterplot(data=sub_df, x=main_weather, y='환자수', hue='월', palette='tab10', 
                    s=80, alpha=0.9, ax=ax2)
    sns.regplot(data=sub_df, x=main_weather, y='환자수', scatter=False, 
                color='black', line_kws={'linestyle':'--', 'linewidth':1.5}, ax=ax2)
    
    ax2.set_title(f"② [비전공자용] 핵심 기상변수({main_weather}) vs 환자수 분포 (월별 색상)", fontsize=12, fontweight='bold')
    ax2.set_xlabel(main_weather)
    ax2.set_ylabel("월별 환자 수 (명)")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(title="월(Month)", bbox_to_anchor=(1.02, 1), loc='upper left')

    # ---------------------------------------------------------
    # [차트 3: 아래 전체] GAM 비선형 스플라인 곡선 (전공자/학술 검증용)
    # ---------------------------------------------------------
    ax3 = fig.add_subplot(2, 1, 2)
    
    if not np.isnan(r2):
        # 3번째 변수(주요 기상변수)의 스플라인 곡선 그리기
        XX = gam.generate_X_grid(term=2)
        pdep, confi = gam.partial_dependence(term=2, X=XX, width=0.95)
        
        ax3.plot(XX[:, 2], pdep, color='crimson', lw=2.5, label='GAM 비선형 추정 곡선')
        ax3.fill_between(XX[:, 2], confi[:, 0], confi[:, 1], color='crimson', alpha=0.15, label='95% 신뢰구간')
        ax3.axhline(0, color='gray', linestyle='--', alpha=0.7)
        
        ax3.set_title(f"③ [전공자/심사위원용] GAM 비선형 변곡점 반응 곡선 (모델 설명력 R² = {r2*100:.1f}%)", 
                      fontsize=12, fontweight='bold')
        ax3.set_xlabel(X_cols[2])
        ax3.set_ylabel("환자수 증감 편차 (Partial Dependence)")
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 이미지 파일 저장 (특수문자 제거)
    safe_name = disease_name.replace('/', '_').replace('\\', '_')
    save_filename = os.path.join(output_dir, f"{code}_{safe_name}_report.png")
    plt.savefig(save_filename, dpi=200, bbox_inches='tight')
    plt.close() # 메모리 해제

# -------------------------------------------------------------
# 5. 전체 결과 요약 파일 저장
# -------------------------------------------------------------
summary_df = pd.DataFrame(summary_results).sort_values(by='GAM_Pseudo_R2', ascending=False)
summary_csv_path = os.path.join(output_dir, 'all_diseases_summary_report.csv')
summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')

print("\n==================================================================")
print(f"🎉 모든 질병 분석 완료!")
print(f" 📁 종합 결과 요약 CSV : {summary_csv_path}")
print(f" 🖼️ 질병별 시각화 그래프 : {output_dir} 폴더 내 리포트 이미지 저장 완료")
print("==================================================================")

# 상위 5개 결과 출력
print("\n📊 [분석 결과 상위 질병 목록]")
print(summary_df.head(7).to_string(index=False))