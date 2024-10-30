import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 포트폴리오 관련 질문과 답변 데이터
questions = [
    "포트폴리오 주제가 무엇인가요?",
    "모델은 어떤걸 썼나요?",
    "인원은 어떻게 되나요?",
    "기간은 어떻게 되나요?",
    "어떤 작업을 진행하였나요?",
    "조장이 누구인가요?",
    "데이터는 어떻게 수집하였나요?",
    "진행하는데 어려움은 없었나요?"
]

answers = [
    "통행약자를 위한 장애물 감지 안전 어시스턴트 서비스입니다.",
    "YOLO 모델을 사용하였습니다.",
    "손주용, 이동근, 박보은, 김연우, 석승연 총 5명 입니다.",
    "약 2주 정도 소요되었습니다.",
    "데이터 수집, 모델 구현, 그 밖에 활용할 기술 구현, 발표 준비, 실시간 감지 시연 등의 작업을 진행하였습니다.",
    "조장은 손주용입니다.",
    "AI허브에 있는 보행자 시선 이미지 데이터와 직접 촬영한 영상과 이미지, 감지하고 싶은 물체를 직접 크롤링하여 수집하였습니다.",
    "많은 데이터를 일일히 라벨링하여 전처리하는게 힘들었지만, 결과가 잘 나온 것 같아서 좋았습니다."
]

# 질문 임베딩과 답변 데이터프레임 생성
question_embeddings = encoder.encode(questions)
df = pd.DataFrame({'question': questions, '챗봇': answers, 'embedding': list(question_embeddings)})

# 대화 이력을 저장하기 위한 Streamlit 상태 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 함수 정의
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산하여 가장 유사한 응답 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

# Streamlit 인터페이스
st.title("통행약자 안전 어시스턴트 서비스 \n 포트폴리오 질문 챗봇")
st.write("저희가 만든 포트폴리오에 대해서 질문해주세요. 예: 주제가 무엇인가요?")

user_input = st.text_input("user", "")

if st.button("Submit"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")