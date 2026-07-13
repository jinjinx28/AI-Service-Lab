import pandas as pd
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 데이터 불러오기
housing = fetch_california_housing(as_frame=True)

df = housing.frame

print(df.head())

# Feature(X): 입력값, Target(y): 정답(예측값) 분리
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Train / Test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# RandomForest 모델 생성
model = RandomForestRegressor(
    random_state=42
)


# 학습
model.fit(X_train, y_train)


# 예측
pred = model.predict(X_test)
print(pred[:10])


# 모델 저장
joblib.dump(
    model,
    "models/house_price_model.pkl"
)