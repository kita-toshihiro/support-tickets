import datetime
import io
import re
from collections import Counter

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="授業アンケート解析", page_icon="📊")
st.title("📊 授業アンケート解析ツール")
st.write(
    """
    CSVファイルをアップロードしてアンケート結果を解析します。
    期待されるCSVのカラム（ヘッダ）は次のとおりです:
    番号,学生番号,アンケート回答,授業が役立ったか（５段階評価）,授業が難しかったか（５段階評価）,回答日時
    """
)

# ファイルアップロード
uploaded_file = st.file_uploader(
    "CSVファイルをアップロードしてください（UTF-8、カンマ区切り）",
    type=["csv"],
    help="ヘッダ行: 番号,学生番号,アンケート回答,授業が役立ったか（５段階評価）,授業が難しかったか（５段階評価）,回答日時",
)

# サンプルCSV（表示用）
sample_csv = """番号,学生番号,アンケート回答,授業が役立ったか（５段階評価）,授業が難しかったか（５段階評価）,回答日時
1,S053,この授業で微積分に対する理解が深まった。特に演習問題が良かった。,5,3,2025-10-25 10:05:12
2,S012,板書が少し早くてついていくのが大変だったが、内容はとてもためになった。,4,4,2025-10-25 10:11:34
3,S076,基本から丁寧に教えてくれて分かりやすかった。応用問題にもっと挑戦したい。,5,2,2025-10-25 10:18:55
4,S009,正直、少し退屈だった。もう少し実生活との関連を説明してほしかった。,2,3,2025-10-25 10:25:01
5,S034,先生の説明が論理的で分かりやすい。数学の楽しさが少し分かった気がする。,5,3,2025-10-25 10:32:40
6,S022,課題の量が多くて負担だったが、その分力がついたと思う。,4,5,2025-10-25 10:38:09
7,S061,グループワークが楽しかった。他の学生と議論することで理解が深まった。,5,3,2025-10-25 10:44:17
8,S002,予習が必須だと感じた。ついていくためにかなり努力が必要だった。,3,5,2025-10-25 10:50:22
9,S073,授業スピードもちょうど良く、重要なポイントが明確だった。,4,2,2025-10-25 10:57:38
10,S030,教科書通りの内容だったが、解説が丁寧で理解しやすかった。,4,3,2025-10-25 11:03:00
"""

def load_csv(file) -> pd.DataFrame:
    # ファイルオブジェクト or str
    if isinstance(file, str):
        buf = io.StringIO(file)
        df = pd.read_csv(buf)
    else:
        # streamlit の UploadedFile はバイナリなので decode
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(io.TextIOWrapper(file, encoding="utf-8"))
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 期待するカラム名の簡易マッチングとリネーム
    col_map = {}
    cols = list(df.columns)
    for c in cols:
        c_norm = re.sub(r"\s+", "", c).lower()
        if "番号" in c or c_norm == "id" or c_norm == "number":
            col_map[c] = "番号"
        elif "学生" in c or "student" in c_norm:
            col_map[c] = "学生番号"
        elif "アンケート" in c or "answer" in c_norm or "comment" in c_norm:
            col_map[c] = "アンケート回答"
        elif "役立" in c or "help" in c_norm:
            col_map[c] = "授業が役立ったか（５段階評価）"
        elif "難し" in c or "difficult" in c_norm:
            col_map[c] = "授業が難しかったか（５段階評価）"
        elif "日時" in c or "date" in c_norm:
            col_map[c] = "回答日時"
    return df.rename(columns=col_map)

def safe_to_numeric(s):
    try:
        return pd.to_numeric(s)
    except Exception:
        return pd.Series(dtype="float64")

def extract_top_words(series: pd.Series, top_n=10):
    # 簡易トークン化：日本語の単語分割はしていないので、ひらがな・カタカナ・漢字の連続を抽出
    texts = series.dropna().astype(str).tolist()
    tokens = []
    for t in texts:
        # 英数字は分ける、記号除去
        t_clean = re.sub(r"[^\w\u3000-\u303F\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]", " ", t)
        # 短すぎる単語を排除
        for w in t_clean.split():
            if len(w) >= 2:
                tokens.append(w)
    counter = Counter(tokens)
    return counter.most_common(top_n)

# データフレーム読み込み（アップロード or サンプル）
if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)
    except Exception as e:
        st.error(f"CSVの読み込みに失敗しました: {e}")
        st.stop()
    st.success("CSVファイルを読み込みました。")
else:
    st.info("CSVをアップロードして解析できます。サンプルデータでプレビューします。")
    df = load_csv(sample_csv)

# 正規化（カラム名を期待する日本語名に）
df = normalize_columns(df)

# 必要なカラムがなければ警告
required_cols = [
    "番号",
    "学生番号",
    "アンケート回答",
    "授業が役立ったか（５段階評価）",
    "授業が難しかったか（５段階評価）",
    "回答日時",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.warning(f"以下の想定カラムが見つかりません: {missing}。可能な限り存在するカラムで解析します。")

# 表示用に先頭を出す
st.header("データプレビュー")
st.dataframe(df.head(50), use_container_width=True)

# 型変換
if "回答日時" in df.columns:
    try:
        df["回答日時"] = pd.to_datetime(df["回答日時"])
    except Exception:
        # 変換失敗は無視
        pass

# 数値カラムを安全に変換
if "授業が役立ったか（５段階評価）" in df.columns:
    df["授業が役立ったか（５段階評価）"] = pd.to_numeric(df["授業が役立ったか（５段階評価）"], errors="coerce")
if "授業が難しかったか（５段階評価）" in df.columns:
    df["授業が難しかったか（５段階評価）"] = pd.to_numeric(df["授業が難しかったか（５段階評価）"], errors="coerce")

# 基本統計
st.header("集計・基本統計")
col1, col2, col3 = st.columns(3)

total_responses = len(df)
col1.metric("回答数", total_responses)

if "授業が役立ったか（５段階評価）" in df.columns:
    avg_useful = df["授業が役立ったか（５段階評価）"].mean(skipna=True)
    col2.metric("授業が役立ったか（平均）", f"{avg_useful:.2f}" if not np.isnan(avg_useful) else "N/A")
else:
    col2.metric("授業が役立ったか（平均）", "N/A")

if "授業が難しかったか（５段階評価）" in df.columns:
    avg_difficulty = df["授業が難しかったか（５段階評価）"].mean(skipna=True)
    col3.metric("授業が難しかったか（平均）", f"{avg_difficulty:.2f}" if not np.isnan(avg_difficulty) else "N/A")
else:
    col3.metric("授業が難しかったか（平均）", "N/A")

# 評価分布のチャート
st.write("")
st.subheader("評価の分布")

charts = []
if "授業が役立ったか（５段階評価）" in df.columns:
    useful_df = df.dropna(subset=["授業が役立ったか（５段階評価）"])
    useful_counts = useful_df["授業が役立ったか（５段階評価）"].value_counts().reset_index()
    useful_counts.columns = ["評価", "件数"]
    useful_counts["評価"] = useful_counts["評価"].astype(str)
    chart1 = alt.Chart(useful_counts).mark_bar().encode(
        x=alt.X("評価:N", title="評価（役立ったか）"),
        y=alt.Y("件数:Q", title="件数"),
        color=alt.Color("評価:N")
    )
    st.altair_chart(chart1, use_container_width=True)
else:
    st.info("「授業が役立ったか（５段階評価）」の列がないため分布を表示できません。")

if "授業が難しかったか（５段階評価）" in df.columns:
    diff_df = df.dropna(subset=["授業が難しかったか（５段階評価）"])
    diff_counts = diff_df["授業が難しかったか（５段階評価）"].value_counts().reset_index()
    diff_counts.columns = ["評価", "件数"]
    diff_counts["評価"] = diff_counts["評価"].astype(str)
    chart2 = alt.Chart(diff_counts).mark_bar().encode(
        x=alt.X("評価:N", title="評価（難しかったか）"),
        y=alt.Y("件数:Q", title="件数"),
        color=alt.Color("評価:N")
    )
    st.altair_chart(chart2, use_container_width=True)
else:
    st.info("「授業が難しかったか（５段階評価）」の列がないため分布を表示できません。")

# テキスト解析：頻出語
st.write("")
st.subheader("アンケート自由記述の頻出語（簡易）")
if "アンケート回答" in df.columns:
    top_words = extract_top_words(df["アンケート回答"].astype(str), top_n=20)
    if top_words:
        top_df = pd.DataFrame(top_words, columns=["語", "出現回数"])
        st.table(top_df.head(20))
    else:
        st.write("十分なテキストがないため頻出語を抽出できませんでした。")
else:
    st.info("「アンケート回答」の列がないため自由記述解析ができません。")

# 時系列解析：日別の回答数
if "回答日時" in df.columns and pd.api.types.is_datetime64_any_dtype(df["回答日時"]):
    st.write("")
    st.subheader("日別の回答数")
    df["回答日"] = df["回答日時"].dt.date
    daily = df.groupby("回答日").size().reset_index(name="件数")
    line = alt.Chart(daily).mark_line(point=True).encode(
        x=alt.X("回答日:T", title="回答日"),
        y=alt.Y("件数:Q", title="件数")
    )
    st.altair_chart(line, use_container_width=True)
else:
    st.info("回答日時の列がない、または日時型に変換できないため日別解析を表示できません。")

st.write("")
st.caption("注: このツールは簡易解析を行います。より高度な自然言語処理や日本語形態素解析を行う場合は MeCab 等を導入してください。")
