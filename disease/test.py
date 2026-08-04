import requests
import json

# 브라우저 미리보기에서 정상 작동했던 완성된 전체 URL
url = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty?serviceKey=6069cd5378ffe531429bdca0ff28ff0f0b0e661007504bc6270cc80995a053b9&returnType=json&numOfRows=100&pageNo=1&stationName=%EC%84%B1%ED%99%A9%EB%8F%99&dataTerm=DAILY&ver=1.0"

# 방화벽 차단 및 타임아웃 방지를 위한 브라우저 위장 헤더
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

print("[정보] 에어코리아 API 요청 중...")

try:
    # timeout을 10초로 지정하여 무한 대기(Time out) 방지
    response = requests.get(url, headers=headers, timeout=10)
    
    print(f"[상태 코드] {response.status_code}")
    
    if response.status_code == 200:
        res_json = response.json()
        print("[성공] API 응답 데이터 수신 완료!")
        
        # 데이터 구조 확인
        items = res_json.get('response', {}).get('body', {}).get('items', [])
        print(f"수집된 측정 데이터 개수: {len(items)}개")
        
        if items:
            print("\n--- 가장 최근 측정 데이터 예시 ---")
            print(f"관측 시간: {items[0].get('dataTime')}")
            print(f"초미세먼지(PM2.5): {items[0].get('pm25Value')} ㎍/㎥")
            print(f"미세먼지(PM10): {items[0].get('pm10Value')} ㎍/㎥")
    else:
        print(f"[실패] 서버 응답 오류: {response.text}")

except requests.exceptions.Timeout:
    print("[오류] 연결 시간 초과 (Read timed out). 공공데이터포털 서버 상태가 불안정하거나 방화벽에 막혔습니다.")
except Exception as e:
    print(f"[오류] 통신 에러 발생: {e}")