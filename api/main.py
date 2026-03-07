from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument
import chromadb
import re
import os
import json
import logging
from datetime import datetime

# .envファイルからOpenAI APIキーなどの環境変数を読み込む
load_dotenv()

# ===== ロガーの設定 =====
# ログフォーマット：時刻・ログレベル・モジュール名・メッセージを出力する
# 例：2026-03-06 10:30:00,123 - INFO - main - ChromaDBを再構築中...
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
# このモジュール専用のロガーを取得する
# __name__を使うことでモジュール名がログに自動的に含まれる
logger = logging.getLogger(__name__)

app = FastAPI(title="Diabetes RAG API")

# ===== パス・モデルの設定 =====
# ChromaDBのSQLiteファイルが格納されるディレクトリ
CHROMA_BASE_DIR = Path("/app/data/chroma")
# 添付文書PDFの格納ディレクトリ
PACKAGE_INSERT_DIR = Path("/app/data/raw/package_inserts")
# ガイドラインPDFの格納ディレクトリ
GUIDELINE_DIR = Path("/app/data/raw/guidelines")

# ベクトル化に使用するEmbeddingsモデル（テキストを数値ベクトルに変換）
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# 回答生成・カテゴリ判定・薬剤名抽出に使用するLLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ===== 対象PDFファイルの定義 =====
# ChromaDBへの登録・再構築時に読み込む添付文書のファイル名リスト
PACKAGE_INSERT_FILES = [
    "metformin.pdf", "glimepiride.pdf", "empagliflozin.pdf",
    "semaglutide.pdf", "sitagliptin.pdf", "linagliptin.pdf",
    "insulin_glargine.pdf", "insulin_aspart.pdf"
    ]
# ChromaDBへの登録・再構築時に読み込むガイドラインのファイル名リスト
GUIDELINE_FILES = ["diabetes_manual_2025.pdf"]

# ===== 検索設定の定義 =====
# 質問文にこれらのキーワードが含まれる場合はガイドライン検索を優先する
GUIDELINE_KEYWORDS = [
    "ガイドライン", "推奨", "第一選択", "治療方針",
    "血糖コントロール", "目標", "標準治療"
    ]

# 質問カテゴリと添付文書セクション番号の対応表
# LLMがカテゴリを判定した後、対応するセクション番号でChromaDBを絞り込む
question_patterns: dict[str, list[int]] = {
    "禁忌": [1, 2, 10],
    "用法用量": [6, 7, 8, 9],
    "副作用_一般":   [11, 11.1, 11.2],
    "副作用_合併症": [9, 11, 11.1, 11.2],
    "腎機能": [2, 7, 9.2, 16],
    "妊婦": [2, 9, 9.5],
    "相互作用": [1, 2, 10],
    "薬物動態": [8, 9, 16],
    "作用機序": [15, 18],
    "薬剤比較": [4, 7, 8, 9, 14, 16, 18],
    }

# LLMのカテゴリ判定プロンプトに含める各パターンの説明文
pattern_descriptions = """
- 禁忌：投与してはいけない患者・条件に関する質問
- 用法用量：投与方法・投与量・投与回数に関する質問
- 副作用_一般：副作用・有害事象そのものを問う質問（例：どんな副作用がありますか）
- 副作用_合併症：腎障害・肝障害・高齢者など特定の背景を持つ患者への副作用リスクを問う質問
- 腎機能：腎機能障害患者への投与・用量調節に関する質問
- 妊婦：妊婦・授乳婦・生殖への影響に関する質問
- 相互作用：他の薬剤との併用・相互作用に関する質問
- 薬物動態：吸収・分布・代謝・排泄に関する質問
- 作用機序：薬効・作用機序に関する質問
- 薬剤比較：複数薬剤の違い・使い分けに関する質問
"""

# セクション分割時に検索対象とする添付文書の見出しキーワード
# 厚労省統一書式の主要セクション名に対応している
target_sections: list[str] = [
    "警告", "禁忌", "組成", "性状", "効能", "効果",
    "用法", "用量", "注意", "相互作用", "副作用",
    "過量投与", "薬物動態", "臨床成績", "薬効", "薬理",
    "有効成分", "取扱い", "包装", "承認"
    ]

# 添付文書のセクション見出し（例：「2. 禁忌」「8．重要な基本的注意」）を
# 検出するための正規表現パターン
# 全角・半角スペースや句点など表記ゆれに対応している
section_pattern = re.compile(
    r'(?:^|\n)\s*'
    r'(\d+(?:\.\d+)?)'
    r'[\.．\s　 ]*'
    r'(?![）\)])'
    r'([^\n\d]{2,30})',
    re.MULTILINE
)


def load_pdfs_with_metadata(
    file_list: list[str],
    doc_type: str,
    pdf_dir: Path
    ) -> list[LCDocument]:
    """PDFファイルを読み込みメタデータを付与する。

    Args:
        file_list: 読み込むPDFファイル名のリスト。
        doc_type: ドキュメントの種別。"package_insert"または"guideline"。
        pdf_dir: PDFファイルが格納されているディレクトリのパス。

    Returns:
        メタデータ付きのDocumentオブジェクトのリスト。
    """
    docs = []
    for fname in file_list:
        fpath = pdf_dir / fname

        # ファイルが存在しない場合はスキップしてWARNINGを出力する
        if not fpath.exists():
            logger.warning("PDFファイルが見つかりません: %s", fpath)
            continue

        # PyMuPDFLoaderで1ページ1Documentとして読み込む
        # PyPDFLoaderではなくPyMuPDFLoaderを使う理由：
        # 日本語PDFのエンコーディング問題に対応しているため
        logger.info("PDFを読み込み中: %s", fname)
        loader = PyMuPDFLoader(str(fpath))
        pages = loader.load()

        # 拡張子を除いたファイル名を薬剤名として使用（例："metformin"）
        drug_name = fname.replace(".pdf", "")

        # 各ページに検索フィルタ用のメタデータを付与する
        for page in pages:
            page.metadata["file_name"] = fname       # ファイル名（参照元表示用）
            page.metadata["doc_type"] = doc_type     # 添付文書orガイドライン
            page.metadata["drug_name"] = drug_name   # 薬剤名フィルタ用
        docs.extend(pages)
        logger.info("%s: %d ページ読み込み完了", fname, len(pages))
    return docs


def split_by_section(
    all_docs: list[LCDocument],
    max_section_size: int = 1000,
    fallback_chunk_size: int = 500,
    fallback_overlap: int = 50
    ) -> list[LCDocument]:
    """厚労省統一形式の見出しを基にセクション単位でチャンク分割する。

    見出しが検出されない場合は文字数ベースのフォールバック分割を適用する。
    セクションサイズがmax_section_sizeを超える場合はさらに細かく分割する。

    Args:
        all_docs: 分割対象のDocumentオブジェクトのリスト。
        max_section_size: セクションの最大文字数。超えた場合は再分割する。
        fallback_chunk_size: フォールバック時のチャンクサイズ（文字数）。
        fallback_overlap: フォールバック時のオーバーラップ文字数。

    Returns:
        セクションメタデータ付きのチャンクリスト。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=fallback_chunk_size,
        chunk_overlap=fallback_overlap
        )
    all_chunks = []

    for doc in all_docs:
        if doc.metadata.get("doc_type") == "guideline":
            chunks = text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
            continue

        text = doc.page_content

        # target_sectionsのキーワードを含み、かつセクション番号が26以下の見出しのみ抽出する
        # 厚労省統一書式の添付文書は最大26セクション構成のため、
        # それより大きい番号は本文中の数値や文献番号として除外する
        matches = [
            m for m in section_pattern.finditer(text)
            if any(target in m.group(2) for target in target_sections)
            and int(m.group(1).split('.')[0]) <= 26
            ]

        if not matches:
            chunks = text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
            continue

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            section_name = f"{match.group(1)}. {match.group(2).strip()}"

            if len(section_text) > max_section_size:
                sub_doc = LCDocument(
                    page_content=section_text,
                    metadata=doc.metadata.copy()
                    )
                sub_chunks = text_splitter.split_documents([sub_doc])
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata["section"] = section_name
                    sub_chunk.metadata["section_number"] = match.group(1)
                all_chunks.extend(sub_chunks)
            else:
                chunk = LCDocument(
                    page_content=section_text,
                    metadata={
                        **doc.metadata,
                        "section": section_name,          # 参照元表示用
                        "section_number": match.group(1)  # 検索フィルタ用
                        }
                    )
                all_chunks.append(chunk)
    return all_chunks


def initialize_vectorstore() -> Chroma:
    """ChromaDBを初期化する。

    既存コレクションにデータが存在する場合はそのまま使用する。
    データが存在しない場合はPDFを読み込んでコレクションを再構築する。

    Returns:
        初期化済みのChromaオブジェクト。
    """
    client = chromadb.PersistentClient(path=str(CHROMA_BASE_DIR))

    # 既存コレクションの件数を確認する
    # ChromaDB 0.5.18ではセッション終了後にデータが消える問題があるため
    # 必ず件数を確認してから使用するかどうかを判断する
    try:
        collection = client.get_collection("diabetes_rag_strategy_c")
        if collection.count() > 0:
            logger.info("既存コレクションを使用: %d チャンク", collection.count())
            return Chroma(
                collection_name="diabetes_rag_strategy_c",
                embedding_function=embeddings,
                persist_directory=str(CHROMA_BASE_DIR)
                )
    except Exception:
        # コレクションが存在しない場合は再構築に進む
        logger.info("既存コレクションが見つかりません。再構築を開始します。")

    # データが存在しない場合はPDFから再構築する
    logger.info("ChromaDBを再構築中...")

    # 添付文書とガイドラインをそれぞれ読み込む
    insert_docs = load_pdfs_with_metadata(
        PACKAGE_INSERT_FILES, "package_insert", PACKAGE_INSERT_DIR
        )
    guideline_docs = load_pdfs_with_metadata(
        GUIDELINE_FILES, "guideline", GUIDELINE_DIR
        )
    all_docs = insert_docs + guideline_docs
    logger.info("PDF読み込み完了: 合計 %d ページ", len(all_docs))

    # セクション単位に分割してpage_contentが空のチャンクを除去する
    section_chunks = split_by_section(all_docs)
    section_chunks = [c for c in section_chunks if c.page_content.strip()]
    logger.info("チャンク分割完了: %d チャンク", len(section_chunks))

    # 既存コレクションが残っている場合は削除してから再登録する
    try:
        client.delete_collection("diabetes_rag_strategy_c")
        logger.info("既存コレクションを削除しました。")
    except Exception:
        pass

    # チャンクをベクトル化してChromaDBに登録する
    # この処理でOpenAI APIを使ってベクトルを生成するため時間がかかる
    logger.info("ベクトル化・ChromaDB登録を開始します（OpenAI APIを使用）...")
    vectorstore = Chroma.from_documents(
        documents=section_chunks,
        embedding=embeddings,
        collection_name="diabetes_rag_strategy_c",
        persist_directory=str(CHROMA_BASE_DIR)
        )
    logger.info("ChromaDB再構築完了: %d チャンク登録済み", vectorstore._collection.count())
    return vectorstore


# APIサーバー起動時にChromaDBを初期化する
# 既存データがあれば使用し、なければPDFから再構築する
vectorstore = initialize_vectorstore()


def detect_sections(question: str) -> tuple[list[int], bool, str]:
    """質問文から該当するセクション番号とガイドライン判定を行う。

    LLMを使って質問パターンを判定し、対応するセクション番号リストを返す。
    ガイドライン関連キーワードが含まれる場合はis_guideline_queryをTrueにする。

    Args:
        question: ユーザーからの質問文。

    Returns:
        以下の3要素のタプル。
        - section_numbers: 該当するセクション番号のリスト。
        - is_guideline_query: ガイドライン質問かどうかのフラグ。
        - response_text: LLMが判定したパターン名のテキスト（カテゴリ表示用）。
    """
    # LLMにパターン一覧と質問を渡してカテゴリを判定させる
    prompt = f"""
以下は医薬品に関する質問パターンの一覧です。
{pattern_descriptions}

次の質問が該当するパターンを全て選んでください。
パターン名のみをカンマ区切りで答えてください。
該当するものがない場合は「該当するパターンはありません。」と答えてください。

質問：{question}

該当するパターン：
"""
    response = llm.invoke(prompt)
    response_text = response.content.strip()

    # LLMの回答にパターン名が含まれているかを確認して
    # 対応するセクション番号リストを構築する
    section_numbers = []
    for pattern_name, numbers in question_patterns.items():
        if pattern_name in response_text:
            # 重複を避けながらセクション番号を追加する
            for num in numbers:
                if num not in section_numbers:
                    section_numbers.append(num)

    # ガイドライン関連キーワードが含まれるかどうかをキーワードマッチで判定する
    # LLMではなくキーワードマッチを使うことでコストと処理時間を節約している
    is_guideline_query = any(kw in question for kw in GUIDELINE_KEYWORDS)

    logger.info(
        "カテゴリ判定結果: [%s] / ガイドライン質問: %s / セクション番号: %s",
        response_text, is_guideline_query, section_numbers
        )
    return section_numbers, is_guideline_query, response_text


def detect_drug_names(question: str) -> list[str]:
    """質問文から薬剤名（ファイル名のstem）を抽出する。

    LLMを使って質問文中の薬剤名を特定し、対応するファイル名を返す。
    薬剤名が特定できない場合は空リストを返す。

    Args:
        question: ユーザーからの質問文。

    Returns:
        薬剤名（ファイル名のstem）のリスト。例: ["metformin", "glimepiride"]
    """
    # 商品名・一般名とファイル名の対応表をプロンプトに含める
    # これにより「ジャヌビア」でも「シタグリプチン」でも同じファイルを特定できる
    drug_map = """
    - メトホルミン、メトグルコ → metformin
    - グリメピリド、アマリール → glimepiride
    - エンパグリフロジン、ジャディアンス → empagliflozin
    - セマグルチド、オゼンピック、リベルサス → semaglutide
    - シタグリプチン、ジャヌビア → sitagliptin
    - リナグリプチン、トラゼンタ → linagliptin
    - インスリン グラルギン、ランタス → insulin_glargine
    - インスリン アスパルト、ノボラピッド → insulin_aspart
    """
    prompt = f"""
以下は薬剤名とファイル名の対応表です。
{drug_map}

次の質問に登場する薬剤のファイル名を全て答えてください。
ファイル名のみをカンマ区切りで答えてください。
特定の薬剤名がない場合は「なし」と答えてください。

質問：{question}

ファイル名：
"""
    response = llm.invoke(prompt)
    response_text = response.content.strip()

    # 薬剤名が特定できなかった場合は空リストを返す
    if "なし" in response_text:
        logger.info("薬剤名: 特定できず")
        return []

    # カンマ区切りのファイル名リストをパースする
    # 全角カンマにも対応するため半角カンマに統一してから分割する
    drug_names = [
        name.strip()
        for name in response_text.replace("、", ",").split(",")
        if name.strip()
        ]
    logger.info("薬剤名の抽出結果: %s", drug_names)
    return drug_names


def retrieve_and_answer(question: str) -> dict:
    """質問に対してRAGで回答を生成する。

    セクション判定・薬剤名抽出・メタデータフィルタリングを組み合わせて
    関連チャンクを取得し、LLMで回答を生成する（戦略D）。
    ガイドライン質問の場合はガイドラインチャンクを優先して取得する。

    Args:
        question: ユーザーからの質問文。

    Returns:
        以下のキーを持つ辞書。
        - answer: LLMが生成した回答テキスト。
        - sources: 参照元情報のリスト。各要素はfile_nameとsectionを持つ辞書。
        - category: LLMが判定した質問カテゴリのテキスト。
    """
    # ステップ1：質問のカテゴリとガイドラインフラグを判定する
    logger.info("RAG処理開始 - 質問: %s", question)
    section_numbers, is_guideline, category = detect_sections(question)

    # ステップ1.5：薬に無関係な質問はここで早期リターンする
    # セクション番号もガイドラインフラグも得られなかった場合は
    # 対象外の質問と判断してLLMを呼び出さずに固定メッセージを返す
    if not section_numbers and not is_guideline:
        logger.warning("対象外の質問として早期リターン: %s", question)
        return {
            "answer": "申し訳ありません。このシステムは糖尿病治療薬の添付文書およびガイドラインに関する質問のみ対応しています。",
            "sources": [],
            "category": category
        }
    
    retrieved_docs = []

    # ステップ2：ガイドライン質問の場合はガイドラインを優先検索する
    # k=3でガイドラインから類似度上位3件を取得する
    if is_guideline:
        guideline_results = vectorstore.similarity_search(
            question, k=3,
            filter={"doc_type": {"$eq": "guideline"}}
            )
        retrieved_docs.extend(guideline_results)

    # ステップ3：セクション番号が特定できた場合は添付文書をセクション絞り込みで検索する
    if section_numbers:
        # 質問に含まれる薬剤名を特定する
        drug_names = detect_drug_names(question)

        for num in section_numbers:
            if drug_names:
                # 薬剤名が特定できた場合はセクション番号×薬剤名でフィルタリングする
                # 組み合わせごとにk=2件ずつ取得する
                # （絞り込みが効いているので2件で十分）
                for drug_name in drug_names:
                    results = vectorstore.similarity_search(
                        question, k=2,
                        filter={
                            "$and": [
                                {"doc_type": {"$eq": "package_insert"}},
                                {"drug_name": {"$eq": drug_name}},
                                {"section_number": {"$eq": str(num)}}
                                ]
                            }
                        )
                    retrieved_docs.extend(results)
            else:
                # 薬剤名が特定できない場合はセクション番号のみでフィルタリングする
                results = vectorstore.similarity_search(
                    question, k=2,
                    filter={
                        "$and": [
                            {"doc_type": {"$eq": "package_insert"}},
                            {"section_number": {"$eq": str(num)}}
                            ]
                        }
                    )
                retrieved_docs.extend(results)

    # ステップ4：セクション絞り込みで0件だった場合のフォールバック検索
    # セクション番号が特定できたが該当チャンクがない場合に
    # 薬剤名・セクションフィルタなしで広く検索する
    if section_numbers and not retrieved_docs:
        logger.warning("セクション絞り込みで0件のためフォールバック検索を実行します。")
        fallback_results = vectorstore.similarity_search(
            question, k=5,
            filter={"doc_type": {"$eq": "package_insert"}}
            )
        retrieved_docs.extend(fallback_results)

    # ステップ5：重複チャンクを除去して上位5件に絞る
    # 同じセクション×薬剤の組み合わせで複数件取得した場合に重複が生じるため
    # page_contentの先頭100文字をキーにして重複を判定する
    seen = set()
    unique_docs = []
    for doc in retrieved_docs:
        content_hash = doc.page_content[:100]
        if content_hash not in seen:
            seen.add(content_hash)
            unique_docs.append(doc)
    # LLMに渡すチャンクは最大5件に制限する
    unique_docs = unique_docs[:5]
    logger.info("取得チャンク数（重複除去・上限適用後）: %d 件", len(unique_docs))

    # ステップ6：取得したチャンクをコンテキストとしてLLMに渡して回答を生成する
    context = "\n\n".join([doc.page_content for doc in unique_docs])
    prompt_text = f"""
あなたは医薬品の専門家です。
以下の医薬品添付文書およびガイドラインの情報をもとに質問に回答してください。
提供された情報に含まれていない内容については「提供された文書には記載がありません」と答えてください。

【参照情報】
{context}

【質問】
{question}

【回答】
"""
    response = llm.invoke(prompt_text)
    answer = response.content

    # ステップ7：参照元情報（ファイル名・セクション名）を重複なしで収集する
    sources = []
    seen_sources: set[str] = set()
    for doc in unique_docs:
        file_name = doc.metadata.get("file_name", "不明")
        section = doc.metadata.get("section", "セクション情報なし")
        key = f"{file_name}_{section}"
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({
                "file_name": file_name,
                "section": section,
                # sectionキーが存在しないチャンクはフォールバック経由のため推定と判定する
                # ChromaDBには保存せず取得後に判定することでスキーマへの影響を避ける
                "section_inherited": "section" not in doc.metadata,
                })

    return {
        "answer": answer,
        "sources": sources,
        "category": category
        }

# ===== Pydanticモデル（リクエスト・レスポンスの型定義） =====

class QuestionRequest(BaseModel):
    """質問リクエストのスキーマ。

    Attributes:
        question: ユーザーからの質問文。
    """

    question: str


class AnswerResponse(BaseModel):
    """回答レスポンスのスキーマ。

    Attributes:
        answer: LLMが生成した回答テキスト。
        sources: 参照元情報のリスト。
        category: LLMが判定した質問カテゴリ。
    """

    answer: str
    sources: list
    category: str


class SaveRequest(BaseModel):
    """保存リクエストのスキーマ。

    Attributes:
        question: 保存対象の質問文。
        answer: 保存対象の回答テキスト。
        sources: 保存対象の参照元情報のリスト。
        category: 保存対象の質問カテゴリ。
        format: 保存形式。"txt"または"json"。デフォルトは"txt"。
    """

    question: str
    answer: str
    sources: list
    category: str
    format: str = "txt"


# ===== エンドポイント =====

@app.get("/health")
def health_check() -> dict:
    """APIサーバーの死活確認エンドポイント。

    Returns:
        statusキーを持つ辞書。正常時は{"status": "ok"}。
    """
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest) -> AnswerResponse:
    """質問を受け取りRAGで回答を生成するエンドポイント。

    Args:
        request: 質問文を含むリクエストオブジェクト。

    Returns:
        回答・参照元・カテゴリを含むレスポンスオブジェクト。

    Raises:
        HTTPException: 質問文が空の場合に400エラーを返す。
    """
    # 空の質問はエラーとして返す
    if not request.question.strip():
        logger.warning("空の質問が送信されました。")
        raise HTTPException(status_code=400, detail="質問を入力してください")

    logger.info("質問を受信: %s", request.question)
    result = retrieve_and_answer(request.question)
    logger.info("回答生成完了")
    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        category=result["category"]
        )


@app.post("/save")
def save_response(request: SaveRequest) -> dict:
    """回答をtxtまたはJSON形式でファイルに保存するエンドポイント。

    保存先は/app/data/saved/ディレクトリ。
    ファイル名はresponse_YYYYMMDD_HHMMSS.txtまたは.json形式。

    Args:
        request: 保存対象のデータと形式を含むリクエストオブジェクト。

    Returns:
        statusとfilenameを含む辞書。
    """
    # 保存先ディレクトリが存在しない場合は作成する
    save_dir = Path("/app/data/saved")
    save_dir.mkdir(parents=True, exist_ok=True)

    # タイムスタンプをファイル名に使用して保存のたびに一意のファイルを生成する
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("保存処理開始 - 形式: %s", request.format)

    if request.format == "json":
        # JSON形式で保存する場合
        filename = f"response_{timestamp}.json"
        content = {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": request.question,
            "category": request.category,
            "answer": request.answer,
            "sources": request.sources
        }
        with open(save_dir / filename, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
    else:
        # txt形式で保存する場合（デフォルト）
        filename = f"response_{timestamp}.txt"
        # 参照元リストを見やすいテキスト形式に変換する
        sources_text = "\n".join([
            f"・{s['file_name']}  {s['section']}"
            for s in request.sources
        ])
        content = f"""========================================
日時：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
========================================

【質問】
{request.question}

【判定カテゴリ】
{request.category}

【回答】
{request.answer}

【参照元】
{sources_text}

========================================
"""
        with open(save_dir / filename, "w", encoding="utf-8") as f:
            f.write(content)

    logger.info("保存完了: %s", save_dir / filename)
    return {"status": "ok", "filename": filename}
