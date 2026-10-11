import requests
import pandas as pd
import html
import re
from nicegui import ui
from sentence_transformers import SentenceTransformer, util
import torch

# ==========================================
# 1. 네이버 API 크롤러 모듈 (대량 수집)
# ==========================================
CLIENT_ID = "25p6ku6h9l"
CLIENT_SECRET = "goJxDPe7SWEbMUHua4WKgoSxYm5IspCm36MLbKhp"

def clean_html(text):
    if not isinstance(text, str):
        return text
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    return text

def fetch_naver_news(query="주식 경제", max_items=200):
    url = "https://naverapihub.apigw.ntruss.com/search/v1/news"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": CLIENT_ID,
        "X-NCP-APIGW-API-KEY": CLIENT_SECRET
    }
    
    all_items = []
    for start_idx in range(1, min(max_items + 1, 1000), 100):
        params = {"query": query, "display": 100, "start": start_idx, "sort": "date"}
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                if not items:
                    break
                for item in items:
                    item['title'] = clean_html(item['title'])
                    item['description'] = clean_html(item['description'])
                all_items.extend(items)
            else:
                break
        except Exception as e:
            print(f"통신 예외: {e}")
            break
            
    print(f"총 {len(all_items)}개의 실시간 뉴스 수집 완료!")
    return pd.DataFrame(all_items)

# ==========================================
# 2. GICS 11대 표준 섹터 정의 및 모던 AI 임베딩 분류기
# ==========================================
GICS_SECTOR_DEFINITIONS = {
    "기술 (IT)": "반도체, 인공지능, AI, 소프트웨어, 하드웨어, 삼성전자, SK하이닉스, 애플, 엔비디아, IT 서비스",
    "헬스케어": "제약, 바이오, 신약 개발, 임상 시험, FDA 승인, 셀트리온, 삼성바이오로직스, 의료기기",
    "금융": "은행, 증권사, 보험, 기준금리 인하 인상, 대출, KB금융, 신한지주, 자산운용, 주가 지수",
    "경기소비재": "자동차, 현대차, 기아, 테슬라, 백화점, 호텔, 면세점, 엔터테인먼트, 하이브, 소비재",
    "필수소비재": "음식료, 농심, CJ제일제당, 담배, KT&G, 생활용품, 화장품, 식료품, 마트",
    "커뮤니케이션서비스": "통신사, SK텔레콤, KT, LG유플러스, 네이버, NAVER, 카카오, KAKAO, 미디어, 콘텐츠",
    "산업재": "조선업, 한화오션, 방산, 한화에어로스페이스, 항공, 건설, 현대건설, 기계 인프라",
    "소재": "화학, 이차전지, 배터리, LG에너지솔루션, 포스코홀딩스, 에코프로, 철강, 리튬, 양극재",
    "에너지": "정유사, S-Oil, SK이노베이션, 석유, 천연가스, 신재생에너지, 유가",
    "유틸리티": "한국전력, 전력 공급, 가스 제조업, 공공요금, 발전소",
    "부동산": "리츠, 상업용 부동산, 임대업, 토지 개발, 주택 시장"
}

print("--- 모던 한국어 AI 임베딩 분류 모델 로딩 중 ---")
embed_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

sector_names = list(GICS_SECTOR_DEFINITIONS.keys())
sector_texts = list(GICS_SECTOR_DEFINITIONS.values())
sector_embeddings = embed_model.encode(sector_texts, convert_to_tensor=True)

def classify_news_to_gics(df):
    if df.empty:
        df['gics_sector'] = "기타"
        return df
    
    docs = (df['title'] + ". " + df['description']).tolist()
    doc_embeddings = embed_model.encode(docs, convert_to_tensor=True)
    cosine_scores = util.cos_sim(doc_embeddings, sector_embeddings)
    
    assigned_sectors = []
    for scores in cosine_scores:
        best_idx = torch.argmax(scores).item()
        assigned_sectors.append(sector_names[best_idx])
        
    df['gics_sector'] = assigned_sectors
    print("--- AI 기반 11대 GICS 섹터 자동 분류 완료 ---")
    return df

def analyze_sentiment(text):
    positive_words = ["최고", "상승", "대박", "돌파", "호조", "성장", "확대", "수혜", "흑자", "회복", "급등", "신고가", "승인"]
    negative_words = ["하락", "감소", "우려", "타격", "적자", "둔화", "폭락", "지연", "손실", "위기", "제재", "부진", "침체"]
    pos_score = sum(1 for w in positive_words if w in text)
    neg_score = sum(1 for w in negative_words if w in text)
    if pos_score > neg_score:
        return "상승", "bg-indigo-50 text-indigo-700 border-indigo-200"
    elif neg_score > pos_score:
        return "하락", "bg-amber-50 text-amber-700 border-amber-200"
    else:
        return "중립", "bg-slate-100 text-slate-700 border-slate-200"

# ==========================================
# 3. 데이터 로드 및 AI 분류 실행
# ==========================================
df_raw = fetch_naver_news(query="주식 경제", max_items=150)
df_classified = classify_news_to_gics(df_raw)

PERSONA_WEIGHT_KEYWORDS = {
    'short': ["급등", "돌파", "속보", "수주", "대박", "폭등", "신고가", "급락", "변동"],
    'swing': ["성장", "확대", "실적", "호조", "개선", "전망", "모멘텀", "추세"],
    'long': ["흑자", "회복", "펀더멘털", "장기", "가치", "배당", "안정", "우수"],
    'conservative': ["우려", "방어", "리스크", "보수", "안전", "주의", "하락", "방어주"]
}

# ==========================================
# 4. NiceGUI 대시보드 UI
# ==========================================
@ui.page('/')
def main_page():
    ui.add_head_html('''
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
            body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #f8fafc; }
            /* ui.select 내부 글자색을 명확한 흰색으로 강제 지정 */
            .q-field__native, .q-field__input, .q-select__focus-target {
                color: #ffffff !important;
            }
            .q-field__marginal {
                color: #94a3b8 !important;
            }
        </style>
    ''')

    with ui.row().classes('w-full min-h-screen no-wrap'):
        
        # [Left Sidebar]
        with ui.column().classes('w-80 bg-slate-900 text-white p-6 justify-between shadow-xl'):
            with ui.column().classes('w-full gap-5'):
                
                # 투자자 성향 선택
                with ui.column().classes('w-full gap-2 mt-4'):
                    ui.label('투자자 성향').classes('text-xs font-semibold text-white tracking-wider')
                    profile_select = ui.select(
                        options={
                            'short': '공격형 단기 트레이더',
                            'swing': '성장주 스윙 투자자',
                            'long': '가치 지향 장기 투자자',
                            'conservative': '보수적 방어형 투자자'
                        },
                        value='short'
                    ).classes('w-full bg-slate-800 text-white rounded-lg p-1 border border-slate-700')

                # GICS 11대 섹터 선택
                sector_options = { "전체 시장": "전체 GICS 시장 보기" }
                for s in sector_names:
                    sector_options[s] = s

                with ui.column().classes('w-full gap-2'):
                    ui.label('관심 분야').classes('text-xs font-semibold text-white tracking-wider')
                    sector_select = ui.select(
                        options=sector_options,
                        value='기술 (IT)'
                    ).classes('w-full bg-slate-800 text-white rounded-lg p-1 border border-slate-700')

            with ui.row().classes('items-center gap-3 p-3 bg-slate-800/50 rounded-xl'):
                ui.avatar('SM', color='indigo', text_color='white').classes('text-sm')
                with ui.column().classes('gap-0'):
                    ui.label('팀 에타 ').classes('text-sm font-semibold text-slate-200')
                    
        # [Right Main Content]
        with ui.column().classes('flex-1 p-8 bg-slate-50 overflow-y-auto'):
            with ui.row().classes('w-full justify-between items-center mb-8'):
                with ui.column().classes('gap-1'):
                    ui.label('추천 뉴스').classes('text-2xl font-bold text-slate-900')
                    ui.label('사용자 투자 성향과 관심 분야에 대해 맞춤형으로 추천된 실시간 뉴스 피드').classes('text-sm text-slate-500')
                
               
            news_container = ui.column().classes('w-full gap-4')

            def update_feed():
                news_container.clear()
                
                selected_profile = profile_select.value
                selected_sector = sector_select.value
                
                if df_classified.empty:
                    with news_container:
                        ui.label('⚠️ 데이터가 없습니다.').classes('text-rose-500 font-semibold')
                    return

                filtered_items = []
                persona_keywords = PERSONA_WEIGHT_KEYWORDS.get(selected_profile, [])
                
                for _, row in df_classified.iterrows():
                    title = row.get('title', '')
                    desc = row.get('description', '')
                    item_sector = row.get('gics_sector', '기타')
                    combined_text = title + " " + desc
                    
                    if selected_sector == '전체 시장' or item_sector == selected_sector:
                        sentiment, badge_style = analyze_sentiment(combined_text)
                        match_score = sum(2 for pk in persona_keywords if pk in combined_text)
                        
                        row_copy = row.copy()
                        row_copy['sentiment'] = sentiment
                        row_copy['badge_style'] = badge_style
                        row_copy['match_score'] = match_score
                        filtered_items.append(row_copy)
                
                filtered_items = sorted(filtered_items, key=lambda x: x['match_score'], reverse=True)
                        
                with news_container:
                    profile_labels = {
                        'short': '단기 트레이더',
                        'swing': '스윙 투자자',
                        'long': '장기 투자자',
                        'conservative': '보수적 투자자'
                    }
                    p_label = profile_labels.get(selected_profile, selected_profile)

                    with ui.row().classes('w-full justify-between items-center bg-white p-4 rounded-2xl shadow-sm border border-slate-100 mb-2'):
                        ui.label(f'현재 선택 섹터: [{selected_sector}] • 총 {len(filtered_items)}건').classes('text-sm font-bold text-slate-700')
                        ui.badge(f'프로필: {p_label}').props('color=indigo').classes('text-xs px-3 py-1 rounded-full')
                    
                    if not filtered_items:
                        with ui.card().classes('w-full p-12 text-center bg-white rounded-2xl shadow-sm border border-slate-100'):
                            ui.label('해당 GICS 섹터로 분류된 뉴스가 없습니다.').classes('text-slate-400 font-medium')
                        return

                    for item in filtered_items:
                        with ui.card().classes('w-full p-6 bg-white rounded-2xl shadow-sm hover:shadow-md transition border border-slate-100'):
                            with ui.row().classes('w-full items-center justify-between mb-2'):
                                with ui.row().classes('gap-2 items-center'):
                                    ui.badge(item['sentiment']).classes(f'text-xs font-bold px-3 py-1 rounded-full border {item["badge_style"]}')
                                    ui.badge(f'섹터: {item["gics_sector"]}').props('color=blue-grey-8').classes('text-xs')
                                    if item['match_score'] > 0:
                                        ui.badge(f'성향 매칭: +{item["match_score"]}').props('color=deep-purple').classes('text-xs')
                                    elif item['match_score'] == 0:
                                        ui.badge('성향 매칭: 기본').props('color=grey-6').classes('text-xs')
                                ui.label(item.get('pubDate', '')).classes('text-xs text-slate-400 font-medium')
                            
                            ui.label(item['title']).classes('text-lg font-bold text-slate-900 tracking-tight mt-1')
                            ui.label(item['description']).classes('text-sm text-slate-600 mt-2 line-clamp-2 leading-relaxed')
                            
                            with ui.row().classes('mt-4 items-center justify-between pt-3 border-t border-slate-100'):
                                ui.label(f'AI 분류 섹터: {item["gics_sector"]}').classes('text-xs text-slate-400 font-semibold')
                                ui.link('원본 기사 읽기 ➔', item['link'], new_tab=True).classes('text-xs font-bold text-indigo-600 hover:text-indigo-800 no-underline')

            profile_select.on_value_change(update_feed)
            sector_select.on_value_change(update_feed)
            update_feed()

ui.run(port=8080)