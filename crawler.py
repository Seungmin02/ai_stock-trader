import requests
import pandas as pd
import html
import re

# 네이버 클라우드 플랫폼 (NAVER API HUB) 인증 정보 고정
CLIENT_ID = "25p6ku6h9l"
CLIENT_SECRET = "goJxDPe7SWEbMUHua4WKgoSxYm5IspCm36MLbKhp"

def clean_html(text):
    if not isinstance(text, str):
        return text
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    return text

def fetch_naver_news(query="주식 경제", max_items=300):
    url = "https://naverapihub.apigw.ntruss.com/search/v1/news"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": CLIENT_ID,
        "X-NCP-APIGW-API-KEY": CLIENT_SECRET
    }
    
    all_items = []
    # 네이버 API는 start 파라미터를 최대 1000까지만 지원하므로 max_items는 1000 이내로 설정
    # 100개씩 끊어서 반복 호출 (Pagination)
    for start_idx in range(1, min(max_items + 1, 1000), 100):
        params = {
            "query": query, 
            "display": 100, 
            "start": start_idx, 
            "sort": "date"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                if not items:
                    break # 더 이상 가져올 뉴스가 없으면 중단
                
                for item in items:
                    item['title'] = clean_html(item['title'])
                    item['description'] = clean_html(item['description'])
                
                all_items.extend(items)
            else:
                print(f"API 에러 발생 (start={start_idx}): {response.text}")
                break
        except Exception as e:
            print(f"통신 예외 발생: {e}")
            break
            
    print(f"총 {len(all_items)}개의 실시간 뉴스 수집 완료!")
    return pd.DataFrame(all_items)