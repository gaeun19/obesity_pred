import streamlit as st
import pandas as pd
import os
import pymysql
from sqlalchemy import create_engine, text
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv #pip install python-dotenv 
from Obesity_pred import create_dataframe, streamlit_data_input, calculate_bmi
from obsity_API import *

HOST = os.getenv("HOST")
USER = os.getenv("USER")
PASSWD = os.getenv("PASSWD")
PORT = os.getenv("PORT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# .env 파일을 읽어서 환경변수로 설정
load_dotenv()  
# 파이썬 전용 데이터베이스 커넥터
pymysql.install_as_MySQLdb()  
# 데이터베이스 연결 엔진
engine = create_engine(f'mysql+pymysql://{USER}:{PASSWD}@{HOST}/MySQL')  



# 1. Gender :  Male, Female
# 2. Age : 연속형
# 3. Height :연속형
# 4. Weight :연속형
# 5. family_history_with_overweight (가족 중 비만 여력 ) : False, True
# 6. FCVC (하루 아채 소비 정도) : 연속형
client = OpenAI()

## streamlit 앱 레이아웃
st.title('비만도 예측 및 운동, 식단 추천')

## sidebar

# Title for the sidebar
st.sidebar.title("정보를 입력해 주세요.")
# 사이드바 스타일링
st.sidebar.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #f5f5f5;
        }
        .sidebar .sidebar-content h2 {
            color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# 1. 성별 선택
user_gender = st.sidebar.radio("성별:", ("남자", "여자"))
# 성별 값을 0.0 또는 1.0으로 변환
if user_gender == '남자':
    c_user_gender = 0.0
else:
    c_user_gender = 1.0

# 2. 나이 입력
user_age = float(st.sidebar.number_input("나이:", min_value=0, max_value=120, value=25))

# 3. 키 입력
user_height = st.sidebar.number_input("키(cm):", min_value=0, max_value=250, value=170)
user_height = float(user_height/100)

# 4. 몸무게 입력
user_weight = float(st.sidebar.number_input("몸무게(kg):", min_value=0, max_value=200, value=70))

# 5. 가족 중 비만 이력 선택
user_family_history = st.sidebar.radio("가족 중 비만 이력 여부:", (False, True))

if user_family_history:
    c_user_family_history =1.0
else:
    c_user_family_history =0.0

# 6. 하루 야채 섭취 횟수 입력
user_fcvc = float(st.sidebar.number_input("하루 야채 섭취 횟수 :", min_value=0, max_value=10, value=3))

# 7. BMI 
user_BMI=calculate_bmi(user_weight,user_height)

# '확인' 버튼
if st.sidebar.button("확인"):
    df_test= create_dataframe(c_user_gender,user_age, user_height, user_weight, c_user_family_history, user_fcvc, user_BMI)
    user_nobeyesdad=streamlit_data_input(df_test)

    
    
    # Display the input values on the main page
    st.write("## 사용자 입력 요약")
    st.write("**성별:**", user_gender)
    st.write("**나이:**", user_age)
    st.write("**키:**", user_height, "cm")
    st.write("**몸무게:**", user_weight, "kg")
    st.write("**가족 중 비만 이력 여부:**", user_family_history)
    st.write("**하루 야채 섭취 횟수:**", user_fcvc)
    st.write("**BMI:**", user_BMI)

    # chat gpt
    nlp_text = st.text("나에게 맞는 운동과 식단을 추천해주세요.")
    FULL_PROMPT = str(table_definition_prompt(description)) + str(nlp_text)
    RESPONSE = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "system",
            "content": f"You are an exercise and diet assistant for me\
                based on the given description definition.\
                given description definition is {table_definition_prompt(description)}\
                The answer should contain description of recommended exercise and diet about my  health state.\
                And you should interpret all the content into Korean.\
                My health status is {user_nobeyesdad}."
            },
            {
                "role": "user",
                "content": f"A query to answer :{FULL_PROMPT}",
            },
        ],
        max_tokens=500,
        temperature=1.0,
        stop=None,
    )

    answer = RESPONSE.choices[0].message.content

    st.write("관리 방법 : ")
    st.write(answer)