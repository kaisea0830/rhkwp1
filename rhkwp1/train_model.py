# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# 데이터 불러오기
df = pd.read_csv("coconut-10-2024.csv")

# 사용될 피처 목록
features = [
    "molecular_weight", "alogp", "topological_polar_surface_area",
    "rotatable_bond_count", "hydrogen_bond_acceptors",
    "hydrogen_bond_donors", "lipinski_rule_of_five_violations",
    "aromatic_rings_count", "formal_charge", "fractioncsp3",
    "number_of_minimal_rings", "van_der_walls_volume"
]

X = df[features]
y = df["qed_drug_likeliness"]

# 모델 학습
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# 모델 저장
joblib.dump(model, "random_forest_qed_model.pkl")

print("✅ 모델이 성공적으로 저장되었습니다!")
