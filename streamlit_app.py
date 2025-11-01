import datetime
import io
import re
from collections import Counter

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="授業アンケート解析（対話式 + Gemini）", page_icon="📊")
st.title("📊 授業アンケート解析ツール（チャット + Gemini対応）")
st.write(
    """
    CSVをアップロードして解析できます。サイドバーのチャットでデータに関する質問が可能です。
    オプションで Google Gemini（Vertex AI / Vertex SDK）を使った自然言語応答を有効にできます。
    （Gemini を使う場合は Google Cloud の設定／認証が必要です — 下の説明を参照してください）
    """
)

# -------------------------
# ファイル読み込み・正規化
# -------------------------
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
    if isinstance(file, str):
        buf = io.StringIO(file)
        df = pd.read_csv(buf)
    else:
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(io.TextIOWrapper(file, encoding="utf-8"))
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    cols = list(df.columns)
    for c in cols:
        c_norm = re.sub(r"\s+", "", str(c)).lower()
        if "番号" in c or c_norm in ("id", "number"):
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

def make_columns_unique(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    seen = {}
    new_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 1
            new_cols.append(c)
        else:
            seen[c] += 1
            new_name = f"{c}({seen[c]-1})"
            while new_name in seen:
                seen[c] += 1
                new_name = f"{c}({seen[c]-1})"
            seen[new_name] = 1
            new_cols.append(new_name)
    df.columns = new_cols
    return df

def extract_top_words(series: pd.Series, top_n=10):
    texts = series.dropna().astype(str).tolist()
    tokens = []
    for t in texts:
        t_clean = re.sub(r"[^\w\u3000-\u303F\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]", " ", t)
        for w in t_clean.split():
            if len(w) >= 2:
                tokens.append(w)
    counter = Counter(tokens)
    return counter.most_common(top_n)

def make_safe_preview(df: pd.DataFrame, n=50) -> pd.DataFrame:
    df_preview = df.head(n).copy()
    for col in df_preview.columns:
        try:
            if df_preview[col].dropna().apply(lambda x: isinstance(x, datetime.date) and not isinstance(x, datetime.datetime)).any():
                df_preview[col] = pd.to_datetime(df_preview[col], errors="coerce")
        except Exception:
            pass
    for col in df_preview.columns:
        try:
            has_bad = df_preview[col].dropna().apply(lambda x: isinstance(x, (list, dict, set, tuple))).any()
        except Exception:
            has_bad = False
        if has_bad:
            df_preview[col] = df_preview[col].astype(str)
    for col in df_preview.select_dtypes(include=["object"]).columns:
        try:
            df_preview[col] = df_preview[col].astype(str)
        except Exception:
            df_preview[col] = df_preview[col].apply(lambda x: str(x) if pd.notna(x) else x)
    return df_preview

# 読み込み
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

# 正規化と一意化
df = normalize_columns(df)
df = make_columns_unique(df)

# 欠けている想定カラムがある場合は注意
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

# DataFrame の型整備
if "回答日時" in df.columns:
    try:
        df["回答日時"] = pd.to_datetime(df["回答日時"], errors="coerce")
    except Exception:
        pass
if "授業が役立ったか（５段階評価）" in df.columns:
    df["授業が役立ったか（５段階評価）"] = pd.to_numeric(df["授業が役立ったか（５段階評価）"], errors="coerce")
if "授業が難しかったか（５段階評価）" in df.columns:
    df["授業が難しかったか（５段階評価）"] = pd.to_numeric(df["授業が難しかったか（５段階評価）"], errors="coerce")

# -------------------------
# Gemini (Vertex AI) 設定（サイドバー）
# -------------------------
st.sidebar.header("LLM（Gemini）設定（オプション）")
use_gemini = st.sidebar.checkbox("Gemini を使った自然言語応答を有効にする", value=False)
st.sidebar.markdown(
    "Gemini を使う場合は、Vertex AI SDK（または google-cloud-aiplatform）を設定してください。\n"
    "このアプリはまずローカルにインストールされた Vertex SDK を試行し、見つからない場合は外部呼び出しの方法を案内します。"
)

# モデル選択（ローカル SDK を使う場合のモデル名のヒント）
gemini_model_hint = st.sidebar.selectbox(
    "モデル（SDK/REST の環境に合わせて選択）",
    options=["gpt-4o-mini", "gemini-proto", "chat-bison@001", "text-bison@001"],
    index=2,
)

# Credential / project info（必要なら）
project = st.sidebar.text_input("GCP プロジェクト（必要な場合）", value="")
location = st.sidebar.text_input("リージョン（例: us-central1）", value="us-central1")

st.sidebar.markdown(
    "注意: Vertex SDK（vertexai / google-cloud-aiplatform）を使う場合はサービスアカウントで認証（環境変数 GOOGLE_APPLICATION_CREDENTIALS 等）が必要です。"
)

# -------------------------
# チャットUI: 一連の対話で質問できる仕組み
# -------------------------
st.sidebar.header("チャット式インターフェース")
st.sidebar.write("ここに質問を入力すると、データフレームに基づいて応答します。Gemini が有効なら LLM を呼び出して自然言語で回答します。")

# 会話履歴をセッションステートで保持
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (user, bot)

# ユーザー入力
user_input = st.sidebar.text_input("質問を入力（例: カラム一覧、サンプル行、'演習' を含む行など）", value="")

# 追加のボタンでよくある質問を挿入できる
if st.sidebar.button("カラム一覧を表示"):
    user_input = "カラム一覧"
if st.sidebar.button("サンプル行を表示"):
    user_input = "サンプル行"
if st.sidebar.button("役立ったかの平均"):
    user_input = "授業が役立ったか（平均）を教えて"
if st.sidebar.button("キーワード検索（アンケート回答）"):
    user_input = "アンケート回答に '演習' を含む行を表示"

# 検索対象列を選ぶ UI（チャット以外でも単独で利用可能）
st.sidebar.markdown("---")
st.sidebar.subheader("列選択（キーワード検索に使用）")
string_cols = list(df.select_dtypes(include=["object", "string"]).columns)
cols_for_search = string_cols if string_cols else list(df.columns)
search_column = st.sidebar.selectbox("検索対象列", options=cols_for_search, index=0 if cols_for_search else None)
st.sidebar.caption("チャットでキーワード検索をしたい場合、ここで列を選んでから質問してください。")

# -------------------------
# Gemini 呼び出しラッパー（可能なら SDK を使う）
# -------------------------
def call_gemini_via_vertex_sdk(prompt: str, model: str = None, max_tokens: int = 512) -> str:
    """
    ローカルに Vertex AI の SDK（vertexai / google-cloud-aiplatform）が入っている場合に実行するラッパー。
    - vertexai.preview.language_models.TextGenerationModel を使う（環境により import 名が変わるため試行します）。
    - ここで例外が出た場合は呼び出しに失敗した旨を文字列で返します。
    """
    try:
        # 新しい vertex-ai SDK（pip install google-cloud-aiplatform か vertexai）
        # Try vertexai (recommended modern SDK)
        try:
            import vertexai
            from vertexai.preview.language_models import TextGenerationModel

            # 初期化は project/location が必要な場合に行う
            if project:
                vertexai.init(project=project, location=location)
            model_name = model or gemini_model_hint or "chat-bison@001"
            tg = TextGenerationModel.from_pretrained(model_name)
            # シンプルな text generation（チャット風にプロンプトをそのまま渡す）
            response = tg.predict(prompt, max_output_tokens=max_tokens)
            return response.text if hasattr(response, "text") else str(response)
        except Exception:
            # Fallback to google-cloud-aiplatform client (older SDK)
            from google.cloud import aiplatform

            if project:
                aiplatform.init(project=project, location=location)
            model_name = model or gemini_model_hint or "chat-bison@001"
            model = aiplatform.TextGenerationModel.from_pretrained(model_name)
            response = model.predict(prompt, max_output_tokens=max_tokens)
            # response が dict のときなどに備える
            if hasattr(response, "text"):
                return response.text
            return str(response)
    except Exception as e:
        return f"[Gemini 呼び出しに失敗しました: {e}]"

def call_gemini(prompt: str) -> str:
    """
    統一インターフェース。Gemini を呼べない場合はエラーメッセージを返す。
    """
    if not use_gemini:
        return "[Gemini 未有効] Gemini を有効にするにはサイドバーのチェックボックスをオンにしてください。"
    # Try SDK-based call
    resp = call_gemini_via_vertex_sdk(prompt, model=gemini_model_hint)
    return resp

# -------------------------
# チャット応答ロジック（ルールベース + Gemini を組み合わせる）
# -------------------------
def answer_query(query: str, df: pd.DataFrame, use_llm: bool = False) -> str:
    """
    query に対してまずシンプルなルールベース応答を作る。
    - ルールで応答が得られない、または LLM を明示的に使う指定がある場合は
      Gemini にプロンプトを送り、DataFrame の簡易コンテキストを含めて応答を得る。
    """
    q = query.strip()
    if q == "":
        return "質問が入力されていません。何か質問してください（例: カラム一覧、サンプル行、アンケートに含まれる特定語の検索など）。"

    q_lower = q.lower()

    # カラム一覧
    if "カラム" in q_lower or "列" in q_lower or "columns" in q_lower:
        return "カラム一覧: " + ", ".join([str(c) for c in df.columns.tolist()])

    # サンプル行
    if "サンプル" in q_lower or "先頭" in q_lower or "head" in q_lower:
        n = 5
        m = re.search(r"(\d+)", q_lower)
        if m:
            n = int(m.group(1))
        preview = df.head(n)
        return f"先頭 {n} 行のプレビュー:\n\n{preview.to_string(index=False)}"

    # 特定カラムの平均値
    if "平均" in q_lower and ("役立" in q_lower or "役に" in q_lower or "useful" in q_lower):
        col = "授業が役立ったか（５段階評価）"
        if col in df.columns:
            avg = df[col].mean(skipna=True)
            return f"'{col}' の平均: {avg:.2f}" if not np.isnan(avg) else f"'{col}' に数値データがありません。"
        else:
            return f"列 '{col}' が見つかりません。カラム一覧を確認してください。"

    # キーワード検索の自然文パターン（例: 'アンケート回答に 演習 を含む行'）
    m = re.search(r"(含む|含める|含まれる).{0,10}['\"“”]?([^'\"、\s]+)['\"”]?", q)
    if m:
        keyword = m.group(2)
        col = search_column if search_column else "アンケート回答"
        if col not in df.columns:
            return f"検索対象の列 '{col}' が見つかりません。代替の列を選ぶか、カラム一覧を確認してください。"
        mask = df[col].astype(str).str.contains(keyword, case=False, na=False)
        matched = df[mask]
        if len(matched) == 0:
            return f"キーワード「{keyword}」に一致する行は見つかりませんでした（列: {col}）。"
        else:
            return f"キーワード「{keyword}」に一致する {len(matched)} 件の行（最大20件表示）:\n\n{matched.head(20).to_string(index=False)}"

    # ルールベースで対応できない、または LLM 指定がある場合は LLM に委ねる
    if use_llm or ("要約" in q_lower or "まとめて" in q_lower or "説明して" in q_lower or "解説" in q_lower):
        # Prepare compact context: カラム一覧 + 上位10行のテキスト（アンケート回答）
        context_lines = []
        context_lines.append("カラム一覧: " + ", ".join([str(c) for c in df.columns.tolist()]))
        if "アンケート回答" in df.columns:
            # include up to first 10 free-text answers for context
            text_sample = df["アンケート回答"].dropna().astype(str).head(10).tolist()
            context_lines.append("アンケート回答のサンプル:")
            for i, t in enumerate(text_sample, 1):
                context_lines.append(f"{i}. {t}")
        # also include basic stats
        stats = []
        if "授業が役立ったか（５段階評価）" in df.columns:
            stats.append(f"役立ったか（平均）: {df['授業が役立ったか（５段階評価）'].mean(skipna=True):.2f}")
        if "授業が難しかったか（５段階評価）" in df.columns:
            stats.append(f"難しかったか（平均）: {df['授業が難しかったか（５段階評価）'].mean(skipna=True):.2f}")
        context_lines.append(" ; ".join(stats))
        system_prompt = (
            "あなたは授業アンケートデータ解析アシスタントです。以下のコンテキストを参照し、ユーザーの質問に日本語でわかりやすく答えてください。"
        )
        full_prompt = system_prompt + "\n\nコンテキスト:\n" + "\n".join(context_lines) + "\n\nユーザーの質問: " + q
        # Call Gemini
        gemini_resp = call_gemini(full_prompt)
        return gemini_resp

    # それ以外は簡易テキスト横断検索
    keyword = q.strip()
    if len(keyword) >= 1:
        text_cols = list(df.select_dtypes(include=["object", "string"]).columns)
        if not text_cols:
            return "テキスト列が見つかりません。具体的にどの列を検索したいか指定してください。"
        mask = pd.Series(False, index=df.index)
        for c in text_cols:
            mask = mask | df[c].astype(str).str.contains(keyword, case=False, na=False)
        matched = df[mask]
        if len(matched) == 0:
            return f"「{keyword}」に一致する行は見つかりませんでした（テキスト列を横断検索）。"
        return f"テキスト列横断検索で {len(matched)} 件ヒット（最大20行表示）:\n\n{matched.head(20).to_string(index=False)}"

    return "すみません、その質問には対応していません。'カラム一覧' や 'サンプル行'、'アンケート回答に 演習 を含む行' などの例を試してください。"

# ユーザーが入力したら処理して履歴に追加
if user_input:
    user_question = user_input.strip()
    st.session_state.chat_history.append(("user", user_question))
    # Gemini を使うかは use_gemini またはクエリ文内の指定で決める（ここはシンプルに use_gemini フラグのみ）
    response = answer_query(user_question, df, use_llm=use_gemini)
    st.session_state.chat_history.append(("bot", response))

# チャット履歴表示
st.subheader("チャット: データについて質問")
for role, text in st.session_state.chat_history[::-1]:
    if role == "user":
        st.markdown(f"**あなた:** {text}")
    else:
        # bot の返答が複数行の場合はコードブロックで表示
        st.markdown(f"**ツール:**\n```\n{text}\n```")

# -------------------------
# 解析パネル（既存機能）
# -------------------------
st.header("解析パネル（表示中データに基づく）")
df_filtered = df.copy()

col1, col2, col3 = st.columns(3)
total_responses = len(df_filtered)
col1.metric("回答数（全体）", total_responses)
if "授業が役立ったか（５段階評価）" in df_filtered.columns:
    avg_useful = df_filtered["授業が役立ったか（５段階評価）"].mean(skipna=True)
    col2.metric("授業が役立ったか（平均）", f"{avg_useful:.2f}" if not np.isnan(avg_useful) else "N/A")
else:
    col2.metric("授業が役立ったか（平均）", "N/A")
if "授業が難しかったか（５段階評価）" in df_filtered.columns:
    avg_diff = df_filtered["授業が難しかったか（５段階評価）"].mean(skipna=True)
    col3.metric("授業が難しかったか（平均）", f"{avg_diff:.2f}" if not np.isnan(avg_diff) else "N/A")
else:
    col3.metric("授業が難しかったか（平均）", "N/A")

st.subheader("データプレビュー（安全化して最大50行）")
df_preview = make_safe_preview(df_filtered, n=50)
try:
    st.dataframe(df_preview, use_container_width=True)
except Exception as e:
    st.warning(f"テーブル表示でエラーが発生しました: {e}")
    fallback = df_preview.copy()
    fallback.columns = [str(c) for c in fallback.columns]
    try:
        st.write(fallback.astype(str))
    except Exception as e2:
        st.error(f"表示に失敗しました: {e2}")
        st.write("列名一覧:", list(df.columns))

st.subheader("評価の分布（表示中データ）")
if "授業が役立ったか（５段階評価）" in df_filtered.columns:
    useful_counts = df_filtered["授業が役立ったか（５段階評価）"].value_counts().reset_index()
    useful_counts.columns = ["評価", "件数"]
    useful_counts["評価"] = useful_counts["評価"].astype(str)
    chart1 = alt.Chart(useful_counts).mark_bar().encode(
        x=alt.X("評価:N", title="評価（役立ったか）"),
        y=alt.Y("件数:Q", title="件数"),
        color=alt.Color("評価:N")
    )
    st.altair_chart(chart1, use_container_width=True)
if "授業が難しかったか（５段階評価）" in df_filtered.columns:
    diff_counts = df_filtered["授業が難しかったか（５段階評価）"].value_counts().reset_index()
    diff_counts.columns = ["評価", "件数"]
    diff_counts["評価"] = diff_counts["評価"].astype(str)
    chart2 = alt.Chart(diff_counts).mark_bar().encode(
        x=alt.X("評価:N", title="評価（難しかったか）"),
        y=alt.Y("件数:Q", title="件数"),
        color=alt.Color("評価:N")
    )
    st.altair_chart(chart2, use_container_width=True)

st.caption(
    "注: Gemini 呼び出しを行うには Vertex AI 関連のライブラリ（vertexai など）を環境にインストールし、"
    "適切な認証（GOOGLE_APPLICATION_CREDENTIALS の設定など）を行ってください。"
)
