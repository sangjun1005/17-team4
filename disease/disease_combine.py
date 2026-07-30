import os
import re
import glob
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

DISEASE_MAP = {
    'J20': '급성 기관지염',
    'J06': '다발성 및 상세불명 부위의 급성 상기도감염',
    'M54': '등통증',
    'M17': '무릎관절증',
    'I10': '본태성(원발성) 고혈압',
    'L23': '알레르기성 접촉피부염',
    'K29': '위염 및 십이지장염',
    'J45': '천식',
    'J30': '혈관운동성 및 알레르기성 비염'
}

def parse_ym(val_str):
    val_clean = re.sub(r'[^0-9]', '', str(val_str))
    if len(val_clean) >= 6:
        year = val_clean[:4]
        month = val_clean[4:6]
        if 1 <= int(month) <= 12:
            return f"{year}-{month.zfill(2)}"
    return None

def parse_file_strictly(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext in ['.xlsx', '.xls']:
            try:
                df = pd.read_excel(file_path, header=None)
            except Exception:
                df = pd.read_csv(file_path, header=None, encoding='cp949')
        else:
            try:
                df = pd.read_csv(file_path, header=None, encoding='utf-8-sig')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, header=None, encoding='cp949')
    except Exception as e:
        print(f"Read error ({os.path.basename(file_path)}): {e}")
        return None

    if df is None or df.empty:
        return None

    month_row_idx = None
    subhead_row_idx = None

    for idx, row in df.iterrows():
        row_str_list = [str(val).strip() for val in row.values if pd.notna(val)]
        if any('진료년월' in s for s in row_str_list) and month_row_idx is None:
            month_row_idx = idx
        if any('환자수' in s for s in row_str_list) and subhead_row_idx is None:
            subhead_row_idx = idx

    if month_row_idx is None or subhead_row_idx is None:
        month_row_idx = 2
        subhead_row_idx = 3

    subhead = [str(x).strip() for x in df.iloc[subhead_row_idx].values]
    month_row = df.iloc[month_row_idx].ffill()

    # 메타데이터 컬럼 위치 감지 (항목/코드, 성별, 연령)
    code_col = 0
    gender_col = 1
    age_col = 2
    
    for i, col_name in enumerate(subhead):
        if any(k in col_name for k in ['코드', '항목', '질병']):
            code_col = i
        elif '성별' in col_name:
            gender_col = i
        elif '연령' in col_name:
            age_col = i

    first_data_col = None
    for i, col_name in enumerate(subhead):
        if '환자수' in col_name:
            first_data_col = i
            break

    if first_data_col is None:
        first_data_col = max(code_col, gender_col, age_col) + 1

    data_df = df.iloc[subhead_row_idx + 1:].copy()

    # 60세 이상 개별 연령대 필터링 (합계/소계 행 제외)
    def is_valid_senior_row(row):
        age_str = str(row[age_col]).strip().replace(' ', '')
        gender_str = str(row[gender_col]).strip()

        if any(x in gender_str for x in ['계', '합계', '소계', '전체']):
            return False
        if any(x in age_str for x in ['계', '합계', '소계', '전체']):
            return False

        return any(k in age_str for k in ['60', '70', '80'])

    valid_mask = data_df.apply(is_valid_senior_row, axis=1)
    data_df = data_df[valid_mask]

    if data_df.empty:
        return None

    file_disease_name = os.path.basename(file_path).split('_')[0].strip()
    records = []

    total_cols = df.shape[1]
    col_idx = first_data_col

    while col_idx < total_cols:
        curr_sub = subhead[col_idx] if col_idx < len(subhead) else ''
        if '환자수' not in curr_sub:
            found = False
            for c in range(col_idx, min(col_idx + 5, total_cols)):
                if '환자수' in subhead[c]:
                    col_idx = c
                    found = True
                    break
            if not found:
                col_idx += 1
                continue

        raw_month = str(month_row[col_idx]).strip()
        ym_str = parse_ym(raw_month)

        if not ym_str:
            col_idx += 5
            continue

        for _, row in data_df.iterrows():
            disease_code = str(row[code_col]).strip()
            gender = str(row[gender_col]).strip()
            age_grp = str(row[age_col]).strip()
            disease_name = DISEASE_MAP.get(disease_code, file_disease_name)

            def parse_num(val):
                try:
                    return float(str(val).replace(',', '').strip())
                except Exception:
                    return 0.0

            patients = parse_num(row[col_idx])
            days = parse_num(row[col_idx + 1]) if col_idx + 1 < total_cols else 0.0
            claims = parse_num(row[col_idx + 2]) if col_idx + 2 < total_cols else 0.0
            cost = parse_num(row[col_idx + 3]) if col_idx + 3 < total_cols else 0.0
            insurer = parse_num(row[col_idx + 4]) if col_idx + 4 < total_cols else 0.0

            records.append({
                '질병코드': disease_code,
                '질병명': disease_name,
                '성별': gender,
                '연령구분': age_grp,
                '진료년월': ym_str,
                '환자수': patients,
                '내원일수': days,
                '청구건수': claims,
                '요양급여비용총액': cost,
                '보험자부담금': insurer
            })

        col_idx += 5

    return pd.DataFrame(records)

def process_all_disease_files(data_folder_path):
    raw_files = glob.glob(os.path.join(data_folder_path, "*.csv")) + \
                glob.glob(os.path.join(data_folder_path, "*.xlsx")) + \
                glob.glob(os.path.join(data_folder_path, "*.xls"))
    
    all_files = [f for f in raw_files if not os.path.basename(f).startswith('~$')]

    print(f"Total disease files found: {len(all_files)}")

    df_list = []
    for fpath in all_files:
        df_parsed = parse_file_strictly(fpath)
        if df_parsed is not None and not df_parsed.empty:
            df_list.append(df_parsed)
            print(f"Parsed successfully: {os.path.basename(fpath)}")

    if not df_list:
        print("Error: 질병 데이터 파싱에 실패했습니다.")
        return None

    full_df = pd.concat(df_list, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=['질병코드', '성별', '연령구분', '진료년월'], keep='first')

    summary_df = full_df.groupby(['질병코드', '질병명', '진료년월'], as_index=False)[
        ['환자수', '내원일수', '청구건수', '요양급여비용총액', '보험자부담금']
    ].sum()
    summary_df = summary_df.sort_values(by=['질병명', '진료년월']).reset_index(drop=True)

    summary_csv = os.path.join(data_folder_path, 'disease_monthly_summary.csv')
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')

    print(f"Summary Completed: {len(summary_df)} rows ({summary_df['진료년월'].min()} ~ {summary_df['진료년월'].max()})")
    return summary_df

if __name__ == '__main__':
    DATA_DIR = './disease_data'
    df_result = process_all_disease_files(DATA_DIR)