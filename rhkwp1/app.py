
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 모델 불러오기
model = joblib.load("random_forest_qed_model.pkl")

# QED 전체 분포 예시 데이터 (실제로는 전체 데이터셋에서 추출)
qed_all_scores = np.random.beta(2, 5, size=1000)  # 예시용 분포

# 등급 분류 함수
def classify_qed(qed):
    if qed >= 0.9:
        return "🟢 매우 우수 (Highly drug-like)"
    elif qed >= 0.7:
        return "🟡 보통 (Moderately drug-like)"
    else:
        return "🔴 낮음 (Poor drug-likeness)"

# 피처 리스트
columns = ["molecular_weight", "alogp", "topological_polar_surface_area",
           "rotatable_bond_count", "hydrogen_bond_acceptors",
           "hydrogen_bond_donors", "lipinski_rule_of_five_violations",
           "aromatic_rings_count", "formal_charge", "fractioncsp3",
           "number_of_minimal_rings", "van_der_walls_volume"]

# 타이틀
st.title("🧪 QED 예측기 (천연물 기반)")
st.markdown("천연물 화합물의 물리화학적 특성을 기반으로 QED 약물 유사성 점수를 예측합니다.")

# 입력 방법 선택
option = st.radio("입력 방법 선택", ["직접 입력", "CSV 업로드"])

# 입력 데이터 준비
if option == "직접 입력":
    input_data = {}
    for col in columns:
        input_data[col] = st.number_input(f"{col}", value=0.0)
    df_input = pd.DataFrame([input_data])
else:
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = None

# 예측 버튼
if st.button("예측하기"):
    if df_input is not None:
        # 예측
        preds = model.predict(df_input)

        # 결과 출력
        st.subheader("📊 예측 결과")
        for i, score in enumerate(preds):
            st.markdown(f"**{i+1}번 샘플 QED 점수:** `{score:.3f}`")
            st.markdown(f"**등급 해석:** {classify_qed(score)}")

        # 히스토그램 시각화
        st.subheader("📈 QED 분포 내 예측값 위치")
        fig, ax = plt.subplots()
        sns.histplot(qed_all_scores, bins=30, kde=True, color="skyblue", ax=ax)
        for score in preds:
            ax.axvline(score, color="red", linestyle="--")
        ax.set_xlabel("QED 점수")
        ax.set_ylabel("빈도")
        ax.set_title("QED 점수 분포와 예측값")
        st.pyplot(fig)

        # 변수 중요도 분석
        st.subheader("🧠 모델 변수 중요도")
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": columns, "Importance": importances})
        imp_df = imp_df.sort_values(by="Importance", ascending=True)

        fig2, ax2 = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=imp_df, ax=ax2)
        ax2.set_title("Feature Importance")
        st.pyplot(fig2)

        # 예측 결과 다운로드
        st.subheader("📁 결과 다운로드")
        df_input["Predicted_QED"] = preds
        csv = df_input.to_csv(index=False).encode("utf-8-sig")
        st.download_button("CSV 다운로드", data=csv, file_name="qed_predictions.csv", mime="text/csv")

    else:
        st.error("❗ 입력값을 먼저 준비해 주세요.")
