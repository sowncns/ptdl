import pandas as pd

def recommend_tour_by_budget_days(
    csv_path,
    places: list,      # danh sách province / place
    max_days: int,
    max_budget: int,
    limit=5
):
    df = pd.read_csv(csv_path)

    # 1. Lọc theo tỉnh/thành (place)
    df = df[df["province"].isin(places)]

    # 2. Lọc theo số ngày
    df = df[df["days"] <= max_days]

    # 3. Lọc theo budget
    df = df[df["price"] <= max_budget]

    if df.empty:
        return None

    # 4. Ưu tiên tour sát budget nhất
    df["score"] = 1 - abs(df["price"] - max_budget) / max_budget

    return df.sort_values("score", ascending=False).head(limit)
