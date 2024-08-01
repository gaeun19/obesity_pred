import joblib
import pandas as pd

# 모델 불러오기
best_model = joblib.load('./best_gb_model.pkl')


def streamlit_data_input(S_test):
    s_pred = best_model.predict(S_test)
    if s_pred == 1:
        return "Normal_Weight"
    elif s_pred == 2:
        return "Obesity_Type_I"
    elif s_pred == 3:
        return "Obesity_Type_II"
    elif s_pred == 4:
        return "Obesity_Type_III"
    elif s_pred == 5:
        return "Overweight_Level_I"
    elif s_pred == 6:
        return "Overweight_Level_II"
    else:
        return "Insufficient_Weight"
    


def create_dataframe(gender, age, height, weight, family_history_with_overweight, fcvc, BMI):
    # 데이터 딕셔너리 생성
    data = {
        "Gender": [gender],
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "family_history_with_overweight": [family_history_with_overweight],
        "FCVC": [fcvc],
        "BMI": [BMI]
 
    }
    
    # pandas DataFrame 생성
    df = pd.DataFrame(data)
    return df


def calculate_bmi(weight,height):
    bmi = weight/(height**2)
    return bmi