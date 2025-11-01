import datetime
import random

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Show app title and description.
st.set_page_config(page_title="サポートチケット", page_icon="🎫")
st.title("🎫 サポートチケット")
st.write(
    """
    このアプリは Streamlit で社内ツールを作る方法を示します。ここではサポートチケットのワークフローを実装しています。
    ユーザーはチケットを作成し、既存のチケットを編集し、統計情報を確認できます。
    """
)

# Create a random Pandas dataframe with existing tickets.
if "df" not in st.session_state:

    # Set seed for reproducibility.
    np.random.seed(42)

    # Make up some fake issue descriptions.
    issue_descriptions = [
        "社内のネットワーク接続の問題",
        "ソフトウェアが起動時にクラッシュする",
        "プリンターが印刷コマンドに応答しない",
        "メールサーバーのダウン",
        "データバックアップの失敗",
        "ログイン認証の問題",
        "ウェブサイトのパフォーマンス低下",
        "セキュリティ脆弱性の検出",
        "サーバールームのハードウェア故障",
        "共有ファイルにアクセスできない従業員",
        "データベース接続の失敗",
        "モバイルアプリがデータを同期しない",
        "VoIP電話システムの問題",
        "リモート社員の VPN 接続問題",
        "システムアップデートによる互換性の問題",
        "ファイルサーバーのストレージ不足",
        "侵入検知システムのアラート",
        "在庫管理システムのエラー",
        "CRM に顧客データが読み込まれない",
        "コラボレーションツールが通知を送信しない",
    ]

    # Generate the dataframe with 100 rows/tickets.
    data = {
        "ID": [f"TICKET-{i}" for i in range(1100, 1000, -1)],
        "Issue": np.random.choice(issue_descriptions, size=100),
        "Status": np.random.choice(["Open", "In Progress", "Closed"], size=100),
        "Priority": np.random.choice(["High", "Medium", "Low"], size=100),
        "Date Submitted": [
            datetime.date(2023, 6, 1) + datetime.timedelta(days=random.randint(0, 182))
            for _ in range(100)
        ],
    }
    df = pd.DataFrame(data)

    # Save the dataframe in session state (a dictionary-like object that persists across
    # page runs). This ensures our data is persisted when the app updates.
    st.session_state.df = df


# Show a section to add a new ticket.
st.header("チケットを追加")

# We're adding tickets via an `st.form` and some input widgets. If widgets are used
# in a form, the app will only rerun once the submit button is pressed.
with st.form("add_ticket_form"):
    issue = st.text_area("問題の説明")
    priority = st.selectbox("優先度", ["High", "Medium", "Low"])
    submitted = st.form_submit_button("送信")

if submitted:
    # Make a dataframe for the new ticket and append it to the dataframe in session
    # state.
    recent_ticket_number = int(max(st.session_state.df.ID).split("-")[1])
    today = datetime.datetime.now().strftime("%m-%d-%Y")
    df_new = pd.DataFrame(
        [
            {
                "ID": f"TICKET-{recent_ticket_number+1}",
                "Issue": issue,
                "Status": "Open",
                "Priority": priority,
                "Date Submitted": today,
            }
        ]
    )

    # Show a little success message.
    st.write("チケットを送信しました！ チケットの詳細：")
    st.dataframe(df_new, use_container_width=True, hide_index=True)
    st.session_state.df = pd.concat([df_new, st.session_state.df], axis=0)

# Show section to view and edit existing tickets in a table.
st.header("既存のチケット")
st.write(f"チケット数: `{len(st.session_state.df)}`")

st.info(
    "セルをダブルクリックするとチケットを編集できます。下のグラフは自動で更新されます。列ヘッダーをクリックして並べ替えることもできます。",
    icon="✍️",
)

# Show the tickets dataframe with `st.data_editor`. This lets the user edit the table
# cells. The edited data is returned as a new dataframe.
edited_df = st.data_editor(
    st.session_state.df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Status": st.column_config.SelectboxColumn(
            "ステータス",
            help="チケットのステータス",
            options=["Open", "In Progress", "Closed"],
            required=True,
        ),
        "Priority": st.column_config.SelectboxColumn(
            "優先度",
            help="チケットの優先度",
            options=["High", "Medium", "Low"],
            required=True,
        ),
    },
    # Disable editing the ID and Date Submitted columns.
    disabled=["ID", "Date Submitted"],
)

# Show some metrics and charts about the ticket.
st.header("統計")

# Show metrics side by side using `st.columns` and `st.metric`.
col1, col2, col3 = st.columns(3)
num_open_tickets = len(st.session_state.df[st.session_state.df.Status == "Open"])
col1.metric(label="オープン中のチケット数", value=num_open_tickets, delta=10)
col2.metric(label="初回対応時間（時間）", value=5.2, delta=-1.5)
col3.metric(label="平均解決時間（時間）", value=16, delta=2)

# Show two Altair charts using `st.altair_chart`.
st.write("")
st.write("##### 月ごとのチケットステータス")
status_plot = (
    alt.Chart(edited_df)
    .mark_bar()
    .encode(
        x="month(Date Submitted):O",
        y="count():Q",
        xOffset="Status:N",
        color="Status:N",
    )
    .configure_legend(
        orient="bottom", titleFontSize=14, labelFontSize=14, titlePadding=5
    )
)
st.altair_chart(status_plot, use_container_width=True, theme="streamlit")

st.write("##### 現在のチケット優先度")
priority_plot = (
    alt.Chart(edited_df)
    .mark_arc()
    .encode(theta="count():Q", color="Priority:N")
    .properties(height=300)
    .configure_legend(
        orient="bottom", titleFontSize=14, labelFontSize=14, titlePadding=5
    )
)
st.altair_chart(priority_plot, use_container_width=True, theme="streamlit")
