# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import plotly.express as px
import plotly.graph_objects as go

# Database
import pymysql 
from sqlalchemy import create_engine, text

# ML
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Others
import warnings
import openai
import os
from dotenv import load_dotenv

import joblib



warnings.filterwarnings("ignore")


# sql 연결 준비
HOST = os.getenv("HOST")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
PORT = os.getenv("PORT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DBAPI = os.getenv("DBAPI")
DATABASE = os.getenv("DATABASE")
load_dotenv() 

# 데이터베이스 연결 설정

# 데이터베이스 URL 생성
DATABASE_URL = f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}/{DATABASE}"

# SQLAlchemy 엔진 생성
engine = create_engine(DATABASE_URL)

# 테이블 목록을 가져오는 쿼리
def get_tables(db_name: str) -> list:
    """Get the list of tables in the database
    Args:
        db_name (str): The name of the database
    Returns:
        list: The list of tables in the database
    """
    with engine.connect() as connection:
        connection.execute(text(f"USE {db_name}"))
        result = connection.execute(text("SHOW TABLES"))
        return [row[0] for row in result]



# CSV 파일 경로
csv_file_path = 'ObesityDataSet.csv'

# CSV 파일을 DataFrame으로 읽기
df = pd.read_csv(csv_file_path)

# DataFrame을 SQL 데이터베이스로 업로드
# # 'your_table_name'을 원하는 테이블 이름으로 변경
# df.to_sql('Obesity', con=engine, index=False)/

# print("CSV 파일이 SQL 데이터베이스로 성공적으로 업로드되었습니다.")
# df = pd.read_csv("C:/ITStudy/04_project/ObesityDataSet.csv")

df = df.drop(columns=["NCP", "CH2O", "FAF","FAVC","CAEC","SMOKE","SCC","TUE","CALC","MTRANS"])

def calculate_bmi(weight,height):
    bmi = weight/(height**2)
    return bmi
df['BMI'] = df.apply(lambda row: calculate_bmi(row['Weight'], row['Height']), axis=1)

# First those that only have 2 possible values with one hot encoder

two_options_features = df[
    ["Gender","family_history_with_overweight"]
]

ct_one_hot = ColumnTransformer(
    [
        (
            "one_hot",
            OneHotEncoder(drop="first"),
            two_options_features.columns,
        ),
    ]
)

encoded_one_hot = pd.DataFrame(
    ct_one_hot.fit_transform(two_options_features), columns=two_options_features.columns
)

df[encoded_one_hot.columns] = encoded_one_hot


# Now the rest with label encoder

NObeyesdad_text = df["NObeyesdad"]

several_options_features = ["NObeyesdad"]

for column in several_options_features:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])


# NObeyesdad 열을 y에 복사
y = df['NObeyesdad'].copy()
# x에서 NObeyesdad 열 삭제
X = df.drop(columns=['NObeyesdad'])

seed = 0

report = pd.DataFrame(columns=["Model", "Accuracy"])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

report = pd.DataFrame(columns=["Model", "Accuracy"])

gb_params = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1, 1],
    "max_depth": [3, 5, 7],
}

random_search = RandomizedSearchCV(GradientBoostingClassifier(), gb_params, random_state=seed).fit(
        X_train, y_train
    )


y_pred = random_search.best_estimator_.predict(X_test)
report.loc[len(report)] = [GradientBoostingClassifier().__str__()[:-2], f1_score(y_test, y_pred, average='micro')]

# print(f"Best params for {GradientBoostingClassifier()}: {random_search.best_params_}")
# print(classification_report(y_test, y_pred))




# # 함수 호출 예시
# gender = 1.0
# age = 30.595632
# height = 1.910672
# weight = 129.232708
# family_history_with_overweight = 1.0
# fcvc = 2.497548
# nobeyesdad = 3

# df = create_dataframe(gender, age, height, weight, family_history_with_overweight, fcvc, nobeyesdad)

# 최적의 모델 저장
joblib.dump(random_search.best_estimator_, './best_gb_model.pkl')



