import streamlit as st
import pandas as pd
from src.predict_label import predict_primary
from src.predict_place import recommend_places
from src.recommend_tour import recommend_tour_by_budget_days
# ======================
# Page config
# ======================
st.set_page_config(
    page_title="Gợi ý du lịch",
    layout="centered"
)

st.title("📋 Gợi ý chuyến du lịch")

# ======================
# Input
# ======================
desc = st.text_input(
    "Mô tả chuyến đi",
    placeholder="VD: thích nghỉ dưỡng yên tĩnh gần biển, không thích leo núi"
)
budget = st.number_input(
    "Ngân sách tối đa (VNĐ)",
    min_value=1000000,
    step=500000,
    value=5000000
)

days = st.slider(
    "Số ngày tối đa",
    min_value=1,
    max_value=7,
    value=3
)
# ======================
# Submit
# ======================
if st.button("📨 Gửi dữ liệu") and desc.strip():

    user_text = desc.strip()

    # ===== 1. Predict primary type (ML)
    primary_preds = predict_primary(user_text)

    st.subheader("🏷️ Loại du lịch gợi ý")
    tag_df = pd.DataFrame(
        primary_preds,
        columns=["Loại du lịch", "Độ phù hợp"]
    )
    tag_df["Độ phù hợp"] = tag_df["Độ phù hợp"].round(3)
    st.table(tag_df)

    # ===== 2. Recommend places
    places = recommend_places(
        csv_path="data/data.csv",
        user_text=user_text,
        limit=5
    )

    st.subheader("📍 Địa điểm phù hợp")
    if places is not None and len(places) > 0:
        places = places.copy()
        places.insert(0, "Xếp hạng", range(1, len(places) + 1))

        st.dataframe(
            places[["Xếp hạng", "province", "place", "description"]],
            hide_index=True,
            width="stretch"
        )
    else:
        st.info("Không tìm thấy địa điểm phù hợp.")

    places_list = places["province"].unique().tolist()
    tours = recommend_tour_by_budget_days(
        csv_path="data/tours_200.csv",
        places=places_list,
        max_days=days,
        max_budget=budget
    )
    st.subheader("🧳 Tour phù hợp ngân sách-Bắt đầu từ TP.HCM")

    if tours is not None:
        st.dataframe(
            tours[["tour_name", "province", "days", "price"]],
            hide_index=True
        )
    else:
        st.info("Không tìm thấy tour phù hợp ngân sách & số ngày.")