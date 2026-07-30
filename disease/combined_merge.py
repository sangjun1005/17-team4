import os
import pandas as pd

DISEASE_FILE = './combined_data/disease_monthly_summary.csv'
WEATHER_FILE = './combined_data/monthly_weather_summary.csv'
AIR_FILE = './combined_data/monthly_air_summary.csv'
OUTPUT_FILE = './combined_data/disease_weather_air_merged.csv'

# 1. 개별 전처리 완료 파일 읽기
disease_df = pd.read_csv(DISEASE_FILE)
weather_df = pd.read_csv(WEATHER_FILE)
air_df = pd.read_csv(AIR_FILE)

# 2. 진료년월 기준 1:1 병합
merged_df = pd.merge(disease_df, weather_df, on='진료년월', how='left')
final_df = pd.merge(merged_df, air_df, on='진료년월', how='left')

# 3. 질병명 및 진료년월 순 정렬
final_df = final_df.sort_values(by=['질병명', '진료년월']).reset_index(drop=True)

# 4. 저장
final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print(f"Final Merged File Saved: {os.path.abspath(OUTPUT_FILE)}")
print(f"Total Records: {len(final_df)} rows")
print(f"Date Range: {final_df['진료년월'].min()} ~ {final_df['진료년월'].max()}")