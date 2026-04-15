import streamlit as st
import ezodf
import json
import io

def process_ods(file_bytes):
    # バイナリストリームとして読み込み
    doc = ezodf.opendoc(io.BytesIO(file_bytes))
    all_sheets_data = {}

    for sheet in doc.sheets:
        sheet_name = sheet.name
        sheet_data = []
        
        # シート内の全行をループ
        for row in sheet.rows():
            row_values = []
            for cell in row:
                # セルの値と数式を取得
                # ezodfでは cell.formula に数式が格納される
                cell_info = {
                    "value": cell.value,
                    "formula": cell.formula if cell.formula else None
                }
                row_values.append(cell_info)
            
            # 空行（すべてのセルが空）でなければ追加
            if any(c["value"] is not None or c["formula"] is not None for c in row_values):
                sheet_data.append(row_values)
        
        all_sheets_data[sheet_name] = sheet_data
        
    return all_sheets_data

# --- Streamlit UI ---
st.set_page_config(page_title="ODS to JSON Converter", layout="wide")
st.title("📊 ODS Sheet Inspector")
st.write("ODSファイルをアップロードすると、全シートの内容を数式付きでJSON表示します。")

uploaded_file = st.file_uploader("ODSファイルを選択してください", type=["ods"])

if uploaded_file is not None:
    try:
        # ファイルの読み込み
        file_bytes = uploaded_file.read()
        
        with st.spinner('解析中...'):
            data = process_ods(file_bytes)
        
        st.success(f"解析完了: {len(data)} 枚のシートが見つかりました")

        # JSONの表示
        st.subheader("JSON Output")
        st.json(data)
        
        # ダウンロードボタン
        json_string = json.dumps(data, indent=4, ensure_ascii=False)
        st.download_button(
            label="JSONをダウンロード",
            data=json_string,
            file_name="spreadsheet_content.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
