"""Evol-Instruct プロンプトテンプレートおよびオペレータ定義モジュール。

Choice / Score / Noul 用の進化生成プロンプト、ハードネガティブ注入制約、
およびデュアルLLMクロスバリデーション用プロンプトを提供する。
"""

from typing import Any

from data.schema import QuestionType
from data.synthetic.taxonomy import SeedSpecification

EVOL_OPERATORS: dict[str, str] = {
    "deepen_constraints": (
        "【制約の深化】状況文（state）に対して、時間制限、契約プランの例外規定、"
        "または複数の前提条件（例: 購入後30日以上経過、有償オプション未加入、他社連携サービスでの発生等）を付加し、"
        "単純なキーワード一致では正解を選べないように複雑化してください。"
    ),
    "add_realism_and_noise": (
        "【リアリティとノイズの付加】状況文（state）に対して、焦りや不満といった口語的・感情的表現、"
        "誤字脱字、またはシステムログ・エラーコード断片を混入させ、"
        "実務の生データに近い揺らぎを再現してください。"
    ),
    "counterfactual_boundary": (
        "【境界事例の生成】正解と次点の選択肢の境界が極めて近接する微妙な状況を構築し、"
        "文脈内のわずか1つの条件の違いによって正解が決定される鋭敏な境界判定シナリオにしてください。"
    ),
}


SYSTEM_PROMPT_GENERATOR = (
    "あなたはエンタープライズ向け判断特化AIの学習データを生成する専門エンジンです。\n"
    "非自己回帰型モデルが学習するために、実務ドメイン（CSトリアージ・障害対応・規約判定）における\n"
    "高品質かつ難易度の高い判断タスクを完全なJSON形式で出力してください。\n"
    "出力は指定されたスキーマに従い、余計な説明文やMarkdownコードブロック装飾（```json ... ```）は含めず、\n"
    "純粋な単一のJSONオブジェクトのみを返してください。"
)


SYSTEM_PROMPT_VALIDATOR = (
    "あなたは厳格なデータセット品質検査官です。\n"
    "提示された状況文（state）、指示（instructions）、および選択肢基準（criteria）のみを根拠に、\n"
    "最も妥当な正解を独立して判定し、問題文に曖昧性や矛盾がないかを検査してください。\n"
    "出力は完全な単一のJSONオブジェクトのみを返してください。"
)


def build_generation_prompt(
    spec: SeedSpecification,
    operator_name: str = "deepen_constraints",
) -> str:
    """シード仕様とEvolオペレータから生成プロンプトを構築する。

    Args:
        spec (SeedSpecification): シード仕様。
        operator_name (str): 適用する進化オペレータ名。

    Returns:
        str: LLM入力用プロンプト文字列。
    """
    operator_instruction = EVOL_OPERATORS.get(
        operator_name, EVOL_OPERATORS["deepen_constraints"]
    )
    domain_info = spec.domain_node

    common_guideline = (
        f"ドメイン分野: {domain_info.domain}\n"
        f"業務カテゴリ: {domain_info.category}\n"
        f"背景事象・制約ノード: {domain_info.constraint_node}\n"
        f"シナリオ詳細: {domain_info.description}\n"
        f"ハードネガティブ誤認要因: {spec.hard_negative_focus}\n"
        f"適用オペレータ:\n{operator_instruction}\n\n"
        "【文字数・構造の制約】\n"
        "- state（状況文）は日本語で 200文字以上 800文字以下 にしてください。\n"
        "- instructions（指示文）は 20文字以上 80文字以下 で、何を判定すべきか端的に記述してください。\n"
        "- 各 criteria（選択肢説明文）は 20文字以上 60文字以下 の簡潔明瞭な文章にしてください。\n"
    )

    if spec.question_type == QuestionType.CHOICE:
        type_specific = (
            f"決定プリミティブ: choice (多肢選択)\n"
            f"候補数: 必ず正確に {spec.num_options} 個 の選択肢を criteria に含めてください。\n"
            "- criteria は {\"key\": \"説明文\"} の辞書形式にしてください。key は英小文字スネークケース（例: 'refund_denied', 'transfer_tech' 等）。\n"
            "- target には criteria に含まれる正解 key を1つ指定してください。\n"
            "- 【重要】正解以外の選択肢（ディストラクター）のうち最低1つは、state のキーワードと強く一致するが、\n"
            "  例外規定や主管部署の違い等の制約により不正解となる「ハードネガティブ」を必ず含めてください。\n"
            "出力JSONスキーマ:\n"
            "{\n"
            '  "question_type": "choice",\n'
            '  "state": "...",\n'
            '  "instructions": "...",\n'
            '  "criteria": {"opt_a": "...", "opt_b": "...", ...},\n'
            '  "target": "opt_a",\n'
            '  "hard_negative_key": "opt_b",\n'
            '  "rationale": "正解の根拠とハードネガティブが不正解である理由"\n'
            "}"
        )
    elif spec.question_type == QuestionType.SCORE:
        type_specific = (
            f"決定プリミティブ: score (順序尺度評価)\n"
            f"段階数: 必ず正確に {spec.num_options} 段階 (0 から {spec.num_options - 1} まで) にしてください。\n"
            f"- criteria のキーは必ず '0', '1', ..., '{spec.num_options - 1}' の昇順連番文字列にしてください。\n"
            "- 各段階の説明文は、深刻度・リスク度・緊急度などの順序尺度として明確に単調推移させてください。\n"
            "- target は対応する正解段階のキー文字列（例: '2'）にしてください。\n"
            "- 境界が近接する状況を作成し、なぜそのスコアであり隣接スコアではないのかの根拠を記述してください。\n"
            "出力JSONスキーマ:\n"
            "{\n"
            '  "question_type": "score",\n'
            '  "state": "...",\n'
            '  "instructions": "...",\n'
            '  "criteria": {"0": "極めて軽微...", "1": "軽微...", "2": "中程度...", ...},\n'
            '  "target": "2",\n'
            '  "rationale": "なぜこのスコアであり、隣接段階ではないかの根拠"\n'
            "}"
        )
    else:  # NOUL
        type_specific = (
            "決定プリミティブ: noul (真偽判定)\n"
            "- criteria は必ず空の辞書 {} にしてください。\n"
            '- target は小文字の "true" または "false" のどちらか一方にしてください。\n'
            "- instructions には、状況文に対して真偽を検証すべき明確な仮説・規約判定文を記述してください。\n"
            "- false の事例では、条件の大半を満たしているが1点だけ反する境界事例を積極的に設計してください。\n"
            "出力JSONスキーマ:\n"
            "{\n"
            '  "question_type": "noul",\n'
            '  "state": "...",\n'
            '  "instructions": "...",\n'
            '  "criteria": {},\n'
            '  "target": "true",\n'
            '  "rationale": "真偽判定の論理的根拠"\n'
            "}"
        )

    return f"{common_guideline}\n{type_specific}"


def build_validation_prompt(sample_dict: dict[str, Any]) -> str:
    """生成サンプルをゼロショットで解かせるクロスバリデーション用プロンプトを構築する。

    Args:
        sample_dict (dict[str, Any]): 生成サンプルの辞書表現。

    Returns:
        str: 検証モデル用入力プロンプト。
    """
    q_type = sample_dict.get("question_type", "choice")
    state = sample_dict.get("state", "")
    instructions = sample_dict.get("instructions", "")
    criteria = sample_dict.get("criteria", {})

    prompt = (
        "以下の状況文、指示文、および選択肢基準のみを論理的に精査し、問いに答えてください。\n\n"
        f"【状況文 (State)】\n{state}\n\n"
        f"【指示文 (Instructions)】\n{instructions}\n\n"
    )

    if q_type == "choice":
        criteria_lines = "\n".join(f"- {k}: {v}" for k, v in criteria.items())
        prompt += (
            f"【選択肢 (Criteria)】\n{criteria_lines}\n\n"
            "タスク: 最も適切と判断されるキー（例: 'opt_a'）を回答してください。\n"
            "また、文脈が曖昧で客観的判断が不可能な場合、または複数選択肢が正解となり得る場合は is_ambiguous を true にしてください。\n"
            "出力JSONスキーマ:\n"
            "{\n"
            '  "predicted_target": "選択したキー文字列",\n'
            '  "is_ambiguous": false,\n'
            '  "confidence": 0.95,\n'
            '  "reasoning": "判断理由"\n'
            "}"
        )
    elif q_type == "score":
        criteria_lines = "\n".join(f"- {k}: {v}" for k, v in criteria.items())
        prompt += (
            f"【評価基準段階 (Criteria)】\n{criteria_lines}\n\n"
            "タスク: 状況文に最も合致する段階キー（'0', '1', 等）を選択してください。\n"
            "判断が曖昧な場合は is_ambiguous を true にしてください。\n"
            "出力JSONスキーマ:\n"
            "{\n"
            '  "predicted_target": "選択した段階キー文字列",\n'
            '  "is_ambiguous": false,\n'
            '  "confidence": 0.95,\n'
            '  "reasoning": "判断理由"\n'
            "}"
        )
    else:  # noul
        prompt += (
            'タスク: 指示文の言明が状況文に対して真であるなら "true"、偽であるなら "false" を選択してください。\n'
            "判断が曖昧な場合は is_ambiguous を true にしてください。\n"
            "出力JSONスキーマ:\n"
            "{\n"
            '  "predicted_target": "true または false",\n'
            '  "is_ambiguous": false,\n'
            '  "confidence": 0.95,\n'
            '  "reasoning": "判断理由"\n'
            "}"
        )

    return prompt
