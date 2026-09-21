"""指示文(Instructions)多様化テンプレートプールモジュール。

固定指示文に対するモデルの過学習を抑止し、ゼロショット追従性を高めるため、
各タスクカテゴリに対して複数の表現・丁寧さ・抽象度の指示文バリエーションを提供する。
"""

import random

# 金融・問い合わせ意図分類向け指示文テンプレート
INTENT_CLASSIFICATION_TEMPLATES: list[str] = [
    "顧客からの問い合わせ内容を分析し、最も適切な対応カテゴリを選択せよ。",
    "以下の問い合わせテキストの主目的・意図に合致する分類を1つ決定せよ。",
    "メッセージの内容を判定し、該当するサポート分類を特定せよ。",
    "発話の意図として最も合致する選択肢を選定せよ。",
    "ユーザー入力の要求内容を分類し、最適な対応項目を選択せよ。",
    "テキストに示された要望を解釈し、該当する手続き種別を決定せよ。",
    "この問い合わせ文が求めているサポート種別を特定せよ。",
]

# ニュース・トピック分類向け指示文テンプレート
TOPIC_CLASSIFICATION_TEMPLATES: list[str] = [
    "記事の主要トピックとして最も合致する分野を選択せよ。",
    "提示されたニューステキストのカテゴリを1つ決定せよ。",
    "本文の内容を評価し、該当する報道分野を特定せよ。",
    "文章の主題として適切なジャンルを選択せよ。",
    "テキストが属するニュースカテゴリを分類せよ。",
    "以下の文章が論じている主要分野を判定せよ。",
]

# 自然言語推論(3値選択)向け指示文テンプレート
NLI_CHOICE_TEMPLATES: list[str] = [
    "前提と仮説の論理的整合性を評価し、適切な関係を選択せよ。",
    "提示された前提文から見て、仮説との論理的関係(含意・中立・矛盾)を判定せよ。",
    "前提事実を踏まえ、仮説が導出可能か、無関係か、あるいは矛盾するかを特定せよ。",
    "前提文の内容に基づき、仮説文に対する含意関係を1つ決定せよ。",
    "2つの文章の論理的含意関係を評価し、適切な分類を選択せよ。",
]

# 真偽判定(Noul型)向け指示文テンプレート
NOUL_VERIFICATION_TEMPLATES: list[str] = [
    "以下の言明は前提事実と照らし合わせて真であるか判定せよ:「{hypothesis}」。",
    "前提テキストの情報のみに基づいて、言明「{hypothesis}」が真実であるか評価せよ。",
    "提示された前提に照らして、言明「{hypothesis}」の真偽を決定せよ。",
    "前提事実から言明「{hypothesis}」が論理的に導かれるか検証せよ。",
    "前提記述と突き合わせ、言明「{hypothesis}」の妥当性を評価せよ。",
]

# 順序尺度・段階評価(Score型)向け指示文テンプレート
SCORE_RATING_TEMPLATES: list[str] = [
    "提示されたテキストの満足度・感情の度合いを指定の段階で評価せよ。",
    "テキスト全体のトーンや印象を分析し、最も合致する評価段階を特定せよ。",
    "レビュー内容を精査し、その評価レベルを基準に従って段階的に判定せよ。",
    "記述された意見や感想の肯定・否定の強さを適切に評価せよ。",
    "提示文の感情極性および温度感を評価基準に沿って判定せよ。",
    "文章のニュアンスを総合的に評価し、該当する段階を選択せよ。",
    "テキストが示す満足度・評価の度合いを、基準に従って1つ決定せよ。",
    "提示された文章の評価度合いを分析し、最適な段階尺度を特定せよ。",
]

TEMPLATES_BY_CATEGORY: dict[str, list[str]] = {
    "intent": INTENT_CLASSIFICATION_TEMPLATES,
    "topic": TOPIC_CLASSIFICATION_TEMPLATES,
    "nli_choice": NLI_CHOICE_TEMPLATES,
    "noul": NOUL_VERIFICATION_TEMPLATES,
    "score": SCORE_RATING_TEMPLATES,
    "rating": SCORE_RATING_TEMPLATES,
}


def sample_instruction(
    category: str,
    rng: random.Random | None = None,
    **kwargs: str,
) -> str:
    """指定カテゴリから指示文テンプレートをランダム抽出して書式化する。

    Args:
        category (str): 指示文カテゴリ ('intent', 'topic', 'nli_choice', 'noul')。
        rng (random.Random | None): 乱数生成器。指定しない場合はデフォルトの random を使用する。
        **kwargs (str): テンプレート埋め込み用パラメータ (例: hypothesis='...')。

    Returns:
        str: 書式化された指示文。

    Raises:
        KeyError: 未定義のカテゴリが指定された場合。
    """
    templates = TEMPLATES_BY_CATEGORY.get(category)
    if templates is None:
        raise KeyError(
            f"未定義の指示文カテゴリです: '{category}'。利用可能: {list(TEMPLATES_BY_CATEGORY.keys())}。"
        )

    chooser = rng.choice if rng is not None else random.choice
    template = chooser(templates)
    return template.format(**kwargs) if kwargs else template
