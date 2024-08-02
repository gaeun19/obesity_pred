# Obesity Prediction & Solution

## Sreamlit Link :

https://obesitypred-6hmr9vbqnfme229nzaow6m.streamlit.app/

## Mini Project (24.08.01 ~ 08.01 : 5 hour)

### With 김승현 배희진 손주연 이가은

- 가은 : Data-preprocess & ML model & Maneger
- 승현 : Data-preprocess & DB
- 희진 : OpenAI
- 주연 : Streamlit

Data Reference

**Obesity or CVD risk (Classify/Regressor/Cluster) by Kaggle**

[https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster/code](https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster/code)

## Story line

[https://drive.google.com/file/d/1gOU1XhUN_u6bQrUER3pP1zjYq_e_UY_p/view?usp=drive_link](https://drive.google.com/file/d/1gOU1XhUN_u6bQrUER3pP1zjYq_e_UY_p/view?usp=drive_link)

## Technical Stack

- ML - scikit-learn, python
- Service - streamlit, OpenAI
- DB - MySQL

## Code

[https://github.com/gaeun19/obesity_pred.git](https://github.com/gaeun19/obesity_pred.git)

## Trouble Shooting

- openai.OpenAIError
    
    ![Untitled](Obesity%20Prediction%20&%20Solution%20b85586ba05e74a7a86e9eeb15e946e3c/Untitled.png)
    
    - 원인 1 : requirments.txt 오류
        - pip freeze >> requirments.txt로 해서 생성
        - 너무 많은 lib 작성 → streamlit에서 오류
    - 해결 : 간략하게 다운받은 것들만 작성
    
    - 원인 2 : .env 파일 OPENAPIKEY  순서 맨 위에 존재
        - .env 파일 읽을 때 순서대로 진행함
    - 해결 :  HOST, …, OPENAPIKEY 순서로 작성할 수 있도록 함.
        - streamlit에서도 secret에 동일하게 작성해야함.
    
- git push 에러
    - 상황 : git push origin main 후 에러 발생
        
        ![Untitled](Obesity%20Prediction%20&%20Solution%20b85586ba05e74a7a86e9eeb15e946e3c/Untitled%201.png)
        
    - 원인 : .env 파일에 OpenApiKey 존재 & .gitignore 파일이 뒤늦게 올라감
        - .gitignore 파일이 정상작동 하지 못함.
    - 해결 → 프로젝트 파일에서 .git 파일 삭제 후 다시 git init 부터 시작함.
- streamlit page loading 시간 오래 걸림(5분 이상)
    - 원인 : streamlit을 띄울 때 데이터 셋에서 모델 학습 & 베스트 웨이트 찾은 후 streamlit load
    - 해결 : 베스트 웨이트 찾은 후 .pkl로 저장
    - 결과 :  streamlit loading 시간 1분 내로 축소

## **Development Plans**
