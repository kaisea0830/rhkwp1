import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

columns = ["molecular_weight", "alogp", "topological_polar_surface_area",
           "rotatable_bond_count", "hydrogen_bond_acceptors",
           "hydrogen_bond_donors", "lipinski_rule_of_five_violations",
           "aromatic_rings_count", "formal_charge", "fractioncsp3",
           "number_of_minimal_rings", "van_der_walls_volume"]

# -------------------- 모델 및 참조 데이터 로딩 --------------------
model = joblib.load("random_forest_qed_model.pkl")
df_reference = pd.read_csv("coconut-10-2024.csv")  # 사전 준비된 참조 데이터
scaler = StandardScaler()
ref_features = df_reference.dropna(subset=[*df_reference.columns])
X_ref_scaled = scaler.fit_transform(ref_features[columns])


qed_all_scores = np.random.beta(2, 5, size=1000)  # QED 분포 예시

def classify_qed(qed):
    if qed >= 0.9:
        return "🟢 매우 우수 (Highly drug-like)"
    elif qed >= 0.7:
        return "🟡 보통 (Moderately drug-like)"
    else:
        return "🔴 낮음 (Poor drug-likeness)"

# -------------------- Streamlit 앱 --------------------
st.title("🧪 QED 예측기 (천연물 기반)")
st.markdown("천연물 화합물의 물리화학적 특성을 기반으로 QED 약물 유사성 점수를 예측합니다.")

option = st.radio("입력 방법 선택", ["직접 입력", "CSV 업로드"])

if option == "직접 입력":
    input_data = {}
    for col in columns:
        input_data[col] = st.number_input(f"{col}", value=0.0)
    df_input = pd.DataFrame([input_data])
else:
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = None

# -------------------- 예측 --------------------
if st.button("예측하기"):
    if df_input is not None:
        preds = model.predict(df_input)

        st.subheader("📊 예측 결과")
        for i, score in enumerate(preds):
            st.markdown(f"<h5>{i+1}번 샘플 QED 점수: <code>{score:.3f}</code></h5>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:14px;'>등급 해석: {classify_qed(score)}</p>", unsafe_allow_html=True)

    # ✅ 기존 존재 여부 확인
            exists = False
            if "canonical_smiles" in df_input.columns and "canonical_smiles" in df_reference.columns:
                exists = df_input.loc[i, "canonical_smiles"] in set(df_reference["canonical_smiles"].dropna())
                existence_text = "✅ 있음" if exists else "❌ 없음"
                st.markdown(f"<p style='font-size:14px;'><b>기존 데이터셋 존재 여부:</b> {existence_text}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='font-size:14px;'><b>기존 데이터셋 존재 여부:</b> (SMILES 없음)</p>", unsafe_allow_html=True)

    # ✅ 유사 화합물 찾기
            try:
                input_scaled = scaler.transform(df_input.iloc[[i]][columns])
                sims = cosine_similarity(input_scaled, X_ref_scaled)[0]
                best_idx = sims.argmax()
                best_row = df_reference.iloc[best_idx]
                best_score = sims[best_idx]

                name = best_row.get("name", "N/A")
                ident = best_row.get("identifier", "N/A")

                st.markdown(
                    f"<p style='font-size:14px;'>가장 유사한 화합물: <b style='color:teal;'>{name}</b> (ID: <code>{ident}</code>)</p>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<p style='font-size:14px;'>유사도 점수: <code>{best_score:.3f}</code></p>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.warning(f"유사 화합물 유사도 비교 오류: {e}")


        # ✅ QED 히스토그램
        st.subheader("📈 QED 분포 내 예측값 위치")
        fig, ax = plt.subplots()
        sns.histplot(qed_all_scores, bins=30, kde=True, color="skyblue", ax=ax)
        for score in preds:
            ax.axvline(score, color="red", linestyle="--")
        ax.set_xlabel("QED 점수")
        ax.set_ylabel("빈도")
        st.pyplot(fig)

        # ✅ Feature Importance
        st.subheader("🧠 모델 변수 중요도")
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": columns, "Importance": importances})
        fig2, ax2 = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=imp_df.sort_values("Importance"), ax=ax2)
        st.pyplot(fig2)

        # ✅ 결과 다운로드
        df_input["Predicted_QED"] = preds
        st.download_button("📁 결과 CSV 다운로드", df_input.to_csv(index=False).encode("utf-8-sig"),
                           file_name="qed_predictions.csv", mime="text/csv")
