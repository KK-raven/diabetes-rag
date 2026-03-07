import streamlit as st
import requests
import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# .envファイルからOpenAI APIキーなどの環境変数を読み込む
load_dotenv()

# ===== ロガーの設定 =====
# ログフォーマット：時刻・ログレベル・モジュール名・メッセージを出力する
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
# このモジュール専用のロガーを取得する
logger = logging.getLogger(__name__)

# ===== 定数 =====
# Docker Compose内のサービス名でFastAPIに接続する
API_BASE_URL = "http://api:8000"

# 保存意図の判定に使用するLLM
# カテゴリ判定・回答生成と同じモデルを使用する
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ===== ページ設定 =====
st.set_page_config(
    page_title="糖尿病治療薬 RAGシステム",
    page_icon="💊",
    layout="wide",
    )


# ===== セッション状態の初期化 =====
# Streamlitはリロードのたびに変数がリセットされるため
# セッションをまたいで保持したいデータはst.session_stateに格納する

# 質問履歴：[{"question": str, "answer": str, "category": str, "sources": list}]
if "history" not in st.session_state:
    st.session_state.history = []

# 現在表示中の回答結果
if "current_result" not in st.session_state:
    st.session_state.current_result = None

# 現在表示中の質問文
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

# テキストエリアのキー管理用（クリアボタンの実装に使用する）
if "input_key" not in st.session_state:
    st.session_state.input_key = 0


def ask_question(question: str) -> dict | None:
    """FastAPIの/askエンドポイントに質問を送信して回答を取得する。

    Args:
        question: ユーザーからの質問文。

    Returns:
        answer・sources・categoryを含む辞書。
        通信エラーやAPIエラーの場合はNoneを返す。
    """
    try:
        logger.info("APIリクエスト送信: %s", question)
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": question},
            # ChromaDB再構築が発生した場合を考慮して長めのタイムアウトを設定する
            timeout=120,
            )
        response.raise_for_status()
        result = response.json()
        logger.info("APIレスポンス受信完了")
        return result
    except requests.exceptions.ConnectionError:
        logger.error("APIサーバーに接続できません: %s", API_BASE_URL)
        st.error("APIサーバーに接続できません。サーバーが起動しているか確認してください。")
        return None
    except requests.exceptions.Timeout:
        logger.error("APIリクエストがタイムアウトしました。")
        st.error("リクエストがタイムアウトしました。しばらく待ってから再試行してください。")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error("APIエラー: %s", e)
        st.error(f"APIエラーが発生しました: {e}")
        return None


def save_response(
    question: str,
    answer: str,
    sources: list,
    category: str,
    fmt: str = "txt",
    ) -> str | None:
    """FastAPIの/saveエンドポイントに回答の保存を依頼する。

    Args:
        question: 保存対象の質問文。
        answer: 保存対象の回答テキスト。
        sources: 保存対象の参照元情報のリスト。
        category: 保存対象の質問カテゴリ。
        fmt: 保存形式。"txt"または"json"。デフォルトは"txt"。

    Returns:
        保存されたファイル名。エラーの場合はNoneを返す。
    """
    try:
        logger.info("保存リクエスト送信 - 形式: %s", fmt)
        response = requests.post(
            f"{API_BASE_URL}/save",
            json={
                "question": question,
                "answer": answer,
                "sources": sources,
                "category": category,
                "format": fmt,
                },
            timeout=30,
            )
        response.raise_for_status()
        filename = response.json().get("filename", "不明")
        logger.info("保存完了: %s", filename)
        return filename
    except requests.exceptions.RequestException as e:
        logger.error("保存リクエストエラー: %s", e)
        st.error(f"保存中にエラーが発生しました: {e}")
        return None


def detect_save_intent(text: str) -> dict:
    """自然言語テキストから保存意図と保存形式を判定する。

    LLMを使ってユーザーの入力が保存を意図しているかどうかを判定する。
    保存を意図している場合はさらに保存形式（txt/json）を判定する。
    形式が明示されていない場合はデフォルトのtxtを返す。

    Args:
        text: ユーザーが入力した自然言語テキスト。

    Returns:
        以下のキーを持つ辞書。
        - save: 保存を意図しているかどうかのフラグ（True/False）。
        - format: 保存形式。"txt"または"json"。
    """
    prompt = f"""
以下のユーザーの発言が「回答を保存したい」という意図を含むかどうかを判定してください。
また保存形式（txtまたはjson）が指定されている場合はその形式も判定してください。

判定ルール：
- 保存・記録・ファイルにしたい・残したい などの表現は保存意図ありと判定する
- 「JSON」「json」「ジェイソン」などの表現があればjson形式と判定する
- 形式の指定がない場合はtxtとする
- 質問や否定の表現は保存意図なしと判定する

【例】
ユーザーの発言：保存して
{{"save": true, "format": "txt"}}

ユーザーの発言：JSONで保存して
{{"save": true, "format": "json"}}

ユーザーの発言：ファイルに残しておいて
{{"save": true, "format": "txt"}}

ユーザーの発言：json形式で記録して
{{"save": true, "format": "json"}}

ユーザーの発言：いや、やっぱりいい
{{"save": false, "format": "txt"}}

ユーザーの発言：メトホルミンの副作用は？
{{"save": false, "format": "txt"}}

ユーザーの発言：{text}

上記の発言に対してJSON形式のみで回答してください。他の文字は含めないでください。
"""
    try:
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        logger.info("保存意図判定結果: %s", response_text)

        # LLMの回答をJSONとしてパースする
        import json
        result = json.loads(response_text)
        return {
            "save": bool(result.get("save", False)),
            "format": result.get("format", "txt"),
            }
    except Exception as e:
        # パースに失敗した場合は保存しない判定にする
        logger.warning("保存意図の判定に失敗しました: %s", e)
        return {"save": False, "format": "txt"}


def render_result(result: dict, question: str) -> None:
    """回答結果を画面に表示する。

    判定カテゴリ・回答・参照元を表示した後に保存ボタンと
    自然言語入力欄を表示する。

    Args:
        result: ask_questionの戻り値（answer・sources・categoryを含む辞書）。
        question: ユーザーからの質問文。
    """
    # 判定カテゴリの表示
    # LLMが質問をどのカテゴリと判断したかをバッジ形式で表示する
    st.markdown("#### 🏷️ 判定カテゴリ")
    # result.get()のデフォルト値"不明"はcategoryキー自体が
    # レスポンスに存在しない場合（APIエラー・通信異常時）にのみ発生する
    # 通常の正常動作では発生しない
    category = result.get("category", "不明")
    if "該当するパターンはありません" in category or category == "不明":
        st.info("カテゴリ：該当なし")
    else:
        # カンマ区切りで複数カテゴリが返る場合があるためバッジとして並べる
        for cat in category.replace("、", ",").split(","):
            cat = cat.strip()
            if cat:
                st.markdown(
                    f'<span style="background-color:#1f6feb;color:white;'
                    f'padding:3px 10px;border-radius:12px;margin:2px;'
                    f'font-size:0.85em;display:inline-block;">{cat}</span>',
                    unsafe_allow_html=True,
                    )
    st.markdown("")

    # 回答の表示
    st.markdown("#### 💬 回答")
    st.markdown(result.get("answer", "回答を取得できませんでした。"))

    # 参照元の表示
    sources = result.get("sources", [])
    if sources:
        st.markdown("#### 📄 参照元")
        for source in sources:
            file_name = source.get("file_name", "不明")
            section = source.get("section", "セクション情報なし")
            inherited = source.get("section_inherited", False)
            # 引き継ぎの場合は（推定）を付けて透明性を確保する
            section_label = f"{section}（推定）" if inherited else section
            st.markdown(f"- `{file_name}` ／ {section_label}")
    else:
        st.markdown("#### 📄 参照元")
        st.caption("参照元の情報がありません。")

    st.divider()

    # ===== 保存機能 =====
    st.markdown("#### 💾 回答の保存")

    # ボタンによる保存（txt形式固定）
    # 回答が表示された後にのみ表示される
    if st.button("この回答をtxtで保存する", type="primary"):
        filename = save_response(
            question=question,
            answer=result["answer"],
            sources=result["sources"],
            category=result["category"],
            fmt="txt",
        )
        if filename:
            st.success(f"保存しました：{filename}")

    # 自然言語による保存（エージェントのデモ）
    # LLMが保存意図と形式を判定してツールを呼び出す
    st.caption(
        "💡 自然言語でも保存できます。"
        "「保存して」「JSONで保存して」など話しかけてください。"
        "（形式を指定しない場合はtxtで保存されます）"
        )
    col1, col2 = st.columns([4, 1])
    with col1:
        natural_input = st.text_input(
            "自然言語で保存を指示する",
            key="natural_save_input",
            placeholder="例：保存して / JSONで保存して / ファイルに残して",
            label_visibility="collapsed",
        )
    with col2:
        # ボタン押下時のみLLMを呼び出すようにする
        natural_submit = st.button("送信", key="natural_submit")

    if natural_input and natural_submit:
        # LLMで保存意図と形式を判定する
        with st.spinner("意図を判定中..."):
            intent = detect_save_intent(natural_input)
        if intent["save"]:
            # 保存意図ありと判定された場合はAPIに保存を依頼する
            fmt = intent["format"]
            logger.info("自然言語保存 - 判定形式: %s", fmt)
            filename = save_response(
                question=question,
                answer=result["answer"],
                sources=result["sources"],
                category=result["category"],
                fmt=fmt,
                )
            if filename:
                st.success(
                    f"保存しました：{filename}（{fmt.upper()}形式）"
                    )
        else:
            # 保存意図なしと判定された場合はその旨を表示する
            st.info("保存の指示として認識されませんでした。")


# ===== サイドバー：質問履歴 =====
with st.sidebar:
    st.markdown("## 📋 質問履歴")
    st.caption("このセッション中の質問一覧です。")

    if not st.session_state.history:
        # 履歴がない場合はメッセージを表示する
        st.caption("まだ質問がありません。")
    else:
        # 新しい質問が上に来るよう逆順で表示する
        for i, item in enumerate(reversed(st.session_state.history)):
            # インデックスは逆順のため全体の長さから調整する
            idx = len(st.session_state.history) - i
            with st.expander(f"Q{idx}：{item['question'][:30]}...", expanded=False):
                st.markdown(f"**カテゴリ：** {item['category']}")
                st.markdown(f"**回答：** {item['answer'][:100]}...")

    # 履歴のクリアボタン
    if st.session_state.history:
        st.divider()
        if st.button("履歴をクリア", type="secondary"):
            st.session_state.history = []
            st.session_state.current_result = None
            st.session_state.current_question = ""
            logger.info("質問履歴をクリアしました。")
            st.rerun()


# ===== メインエリア =====
st.title("💊 糖尿病治療薬 RAGシステム")
st.caption(
    "医薬品添付文書・診療ガイドラインに基づいて質問に回答します。"
    "本システムの回答は参考情報であり、医療上の判断は専門家にご相談ください。"
    )
st.divider()

# 質問入力エリア
st.markdown("#### ❓ 質問を入力してください")
question_input = st.text_area(
    "質問",
    placeholder=(
        "例：メトホルミンの禁忌を教えてください\n"
        "例：腎機能障害患者にエンパグリフロジンは使えますか？"
        ),
    height=120,
    label_visibility="collapsed",
    # キーを変えることでStreamlitにウィジェットの再生成を強制しクリアを実現する
    key=f"question_input_{st.session_state.input_key}",
    )

col_submit, col_clear = st.columns([5, 1])
with col_submit:
    submit = st.button("質問する", type="primary", use_container_width=True)
with col_clear:
    if st.button("クリア", use_container_width=True):
        # キーをインクリメントしてテキストエリアをリセットする
        st.session_state.input_key += 1
        # 画面の回答表示もクリアする（履歴への保存は送信時に完了済み）
        st.session_state.current_result = None
        st.session_state.current_question = ""
        st.rerun()

if submit:
    if not question_input.strip():
        st.warning("質問を入力してください。")
    else:
        with st.spinner("回答を生成中...（初回起動時はChromaDBの構築に数分かかる場合があります）"):
            result = ask_question(question_input)

        if result:
            st.session_state.current_result = result
            st.session_state.current_question = question_input
            # 新しい質問送信時に自然言語保存入力をリセットする
            st.session_state.natural_save_input = ""
            st.session_state.history.append({
                "question": question_input,
                "answer": result.get("answer", ""),
                "category": result.get("category", ""),
                "sources": result.get("sources", []),
                })
            logger.info("質問履歴に追加: %s", question_input)

# 回答結果の表示
# セッション状態に回答が存在する場合のみ表示する
if st.session_state.current_result:
    st.divider()
    render_result(
        result=st.session_state.current_result,
        question=st.session_state.current_question,
        )
