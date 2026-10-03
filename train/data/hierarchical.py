"""粗密階層マッピングおよびオントロジー定義モジュール。

大分類 (Coarse) と細分類 (Fine) のオントロジー定義、双方向インデックス、
および MECE (相互排他的かつ網羅的) 性のバリデーションを提供する。
Rust ランタイム側 (crates/sokuto-runtime/src/hierarchical/mapping.rs) と
完全な互換性を保持する。
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_NEGATIVE_KEYS = [
    "other",
    "none",
    "unknown",
    "other_or_unsupported",
    "該当なし",
    "その他",
    "不明",
]


@dataclass
class HierarchicalMapping:
    """大分類と細分類の対応関係を保持する階層マッピング。

    大分類 (4〜8 クラスタ) と細分類 (各 8〜12 候補) の双方向高速参照を実現する。

    Attributes:
        name (str): マッピング全体の識別名。
        coarse_categories (dict[str, str]): 大分類識別子と説明文のマッピング。
        fine_criteria (dict[str, str]): 全細分類識別子と説明文のマッピング。
        coarse_to_fine (dict[str, list[str]]): 大分類識別子から配下の細分類識別子リストへのマッピング。
        fine_to_coarse (dict[str, str]): 細分類識別子から所属する大分類識別子への逆引きマッピング。
        all_coarse_keys (list[str]): 事前キャッシュされた全大分類キーの順序保持リスト。
        all_fine_keys (list[str]): 事前キャッシュされた全細分類キーの順序保持リスト。
        negative_keys (list[str]): 受け皿 (None / その他) 候補として判定するキー一覧。
    """

    name: str
    coarse_categories: dict[str, str] = field(default_factory=dict)
    fine_criteria: dict[str, str] = field(default_factory=dict)
    coarse_to_fine: dict[str, list[str]] = field(default_factory=dict)
    fine_to_coarse: dict[str, str] = field(default_factory=dict)
    all_coarse_keys: list[str] = field(default_factory=list)
    all_fine_keys: list[str] = field(default_factory=list)
    negative_keys: list[str] = field(
        default_factory=lambda: list(DEFAULT_NEGATIVE_KEYS)
    )

    def validate(self) -> None:
        """マッピングの MECE 性および整合性を検証する。

        Raises:
            ValueError: 大分類または細分類が空、MECE 違反、または双方向不一致の場合。
        """
        if not self.coarse_categories:
            raise ValueError("大分類 (coarse_categories) が 1 つも定義されていません。")

        if not self.fine_criteria:
            raise ValueError("細分類 (fine_criteria) が空です。")

        # 1. 各大分類の重複および MECE 性検証
        seen_fine_keys: set[str] = set()
        for coarse_key, fine_list in self.coarse_to_fine.items():
            for fine_key in fine_list:
                if fine_key in seen_fine_keys:
                    raise ValueError(
                        f"細分類 `{fine_key}` が複数の大分類に重複して割り当てられています (MECE 違反)。"
                    )
                seen_fine_keys.add(fine_key)

        # 2. 大分類配下の細分類検証
        for coarse_key, coarse_desc in self.coarse_categories.items():
            if not coarse_key.strip():
                raise ValueError("空の大分類キーが存在します。")
            if not coarse_desc.strip():
                raise ValueError(f"大分類 `{coarse_key}` の説明文が空です。")

            fine_list = self.coarse_to_fine.get(coarse_key)
            if fine_list is None:
                raise ValueError(
                    f"大分類 `{coarse_key}` に対する細分類リストが定義されていません。"
                )
            if not fine_list:
                raise ValueError(f"大分類 `{coarse_key}` 配下の細分類リストが空です。")

            for fine_key in fine_list:
                if fine_key not in self.fine_criteria:
                    raise ValueError(
                        f"大分類 `{coarse_key}` に指定された細分類 `{fine_key}` が fine_criteria に存在しません。"
                    )

        # 3. 逆引きマッピングの整合性検証
        for fine_key, coarse_key in self.fine_to_coarse.items():
            if fine_key not in self.fine_criteria:
                raise ValueError(
                    f"逆引きに存在する細分類 `{fine_key}` が fine_criteria に未定義です。"
                )
            if coarse_key not in self.coarse_categories:
                raise ValueError(
                    f"細分類 `{fine_key}` の所属先大分類 `{coarse_key}` が coarse_categories に未定義です。"
                )
            expected_fine_list = self.coarse_to_fine.get(coarse_key, [])
            if fine_key not in expected_fine_list:
                raise ValueError(
                    f"双方向不一致: 細分類 `{fine_key}` は大分類 `{coarse_key}` に割り当てられていますが、coarse_to_fine に含まれていません。"
                )

        # 4. 全細分類の網羅性検証
        for fine_key in self.fine_criteria:
            if fine_key not in self.fine_to_coarse:
                raise ValueError(
                    f"細分類 `{fine_key}` がどの親大分類にも割り当てられていません。"
                )

    def to_dict(self) -> dict[str, Any]:
        """辞書オブジェクトへ変換する。

        Returns:
            dict[str, Any]: シリアライズ可能な辞書表現。
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HierarchicalMapping":
        """辞書オブジェクトから HierarchicalMapping を復元する。

        Args:
            data (dict[str, Any]): 辞書データ。

        Returns:
            HierarchicalMapping: 復元されたマッピングインスタンス。
        """
        all_coarse_keys = list(
            data.get("all_coarse_keys")
            or list(data.get("coarse_categories", {}).keys())
        )
        all_fine_keys = list(
            data.get("all_fine_keys") or list(data.get("fine_criteria", {}).keys())
        )
        mapping = cls(
            name=data["name"],
            coarse_categories=dict(data.get("coarse_categories", {})),
            fine_criteria=dict(data.get("fine_criteria", {})),
            coarse_to_fine={
                k: list(v) for k, v in data.get("coarse_to_fine", {}).items()
            },
            fine_to_coarse=dict(data.get("fine_to_coarse", {})),
            all_coarse_keys=all_coarse_keys,
            all_fine_keys=all_fine_keys,
            negative_keys=list(data.get("negative_keys", DEFAULT_NEGATIVE_KEYS)),
        )
        mapping.validate()
        return mapping

    def to_json(self, file_path: str | Path) -> None:
        """JSON ファイルへシリアライズして保存する。

        Args:
            file_path (str | Path): 出力先ファイルパス。
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, file_path: str | Path) -> "HierarchicalMapping":
        """JSON ファイルから HierarchicalMapping を読み込んで復元する。

        Args:
            file_path (str | Path): 入力 JSON ファイルパス。

        Returns:
            HierarchicalMapping: 復元されたマッピングインスタンス。
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def banking77(cls) -> "HierarchicalMapping":
        """Banking77 向け 7 大分類プリセットを構築する。

        77 候補の金融インテントを 7 個の大分類クラスタへ網羅的に分類する。
        Rust 側 HierarchicalMapping::banking77() と完全一致する。

        Returns:
            HierarchicalMapping: Banking77 プリセットインスタンス。
        """
        categories: list[tuple[str, str, list[tuple[str, str]]]] = [
            (
                "card_ops",
                "カード発行・配送・更新・有効化および利用設定",
                [
                    ("activate_my_card", "新しいカードを有効化・利用開始登録したい。"),
                    (
                        "card_about_to_expire",
                        "カードの有効期限が近づいているため更新したい。",
                    ),
                    (
                        "card_acceptance",
                        "カードが加盟店や決済端末で利用可能か確認したい。",
                    ),
                    (
                        "card_arrival",
                        "申し込んだカードがいつ届くか、配送状況を確認したい。",
                    ),
                    (
                        "card_delivery_estimate",
                        "カードの発送時期や到着予定日を知りたい。",
                    ),
                    (
                        "card_linking",
                        "カードをアプリや外部アカウントに連携・登録したい。",
                    ),
                    ("card_not_working", "カード決済や読み取りが正常に動作しない。"),
                    ("card_swallowed", "ATM や端末にカードが吸い込まれて戻らない。"),
                    (
                        "getting_spare_card",
                        "予備カードや追加カードの発行を申請したい。",
                    ),
                    (
                        "getting_virtual_card",
                        "即時利用可能なバーチャルカードを発行したい。",
                    ),
                    (
                        "order_physical_card",
                        "プラスチック製の物理カードを新規発行したい。",
                    ),
                    (
                        "pin_blocked",
                        "PIN コードの入力間違い等で暗証番号がロックされた。",
                    ),
                ],
            ),
            (
                "transfers",
                "口座間送金・銀行振込・送金エラーおよび送金取消",
                [
                    (
                        "balance_not_updated_after_bank_transfer",
                        "銀行振込を行ったが残高に反映されない。",
                    ),
                    (
                        "beneficiary_not_allowed",
                        "振込先口座が登録できない、または送金不可と表示される。",
                    ),
                    (
                        "cancel_transfer",
                        "実行済みまたは予約中の送金・振込を取り消したい。",
                    ),
                    ("failed_transfer", "送金手続きそのものが失敗・エラーになった。"),
                    (
                        "pending_transfer",
                        "送金処理が保留中 (Pending) のまま完了しない。",
                    ),
                    ("transfer_fee_charged", "送金時に想定外の手数料が差し引かれた。"),
                    (
                        "transfer_into_account",
                        "外部口座から本口座への振込方法や着金を知りたい。",
                    ),
                    (
                        "transfer_not_received_by_recipient",
                        "送金完了通知が出たが受取人に届いていない。",
                    ),
                    (
                        "transfer_timing",
                        "送金が相手に届くまでの所要時間やスケジュールを知りたい。",
                    ),
                    (
                        "unable_to_verify_identity_for_transfer",
                        "送金時の本人確認ステップを通過できない。",
                    ),
                    (
                        "wrong_amount_sent",
                        "振込金額を誤って多く（または少なく）送金してしまった。",
                    ),
                ],
            ),
            (
                "payments",
                "店舗・オンライン決済・カード決済手数料および二重請求",
                [
                    (
                        "card_payment_fee_charged",
                        "カード決済時に余分な手数料が請求された。",
                    ),
                    (
                        "card_payment_not_recognised",
                        "カードの利用明細に身に覚えのない請求がある。",
                    ),
                    (
                        "card_payment_wrong_exchange_rate",
                        "海外決済時の為替レートが想定と異なる。",
                    ),
                    (
                        "declined_card_payment",
                        "店舗またはオンラインでのカード決済が拒否された。",
                    ),
                    (
                        "direct_debit_payment_problem",
                        "口座引き落とし・自動引き落としの支払いに失敗した。",
                    ),
                    (
                        "disposable_card_limits",
                        "使い捨てバーチャルカードの利用限度額や制限を知りたい。",
                    ),
                    (
                        "extra_charge_on_statement",
                        "請求明細に身に覚えのない追加料金が加算されている。",
                    ),
                    (
                        "pending_card_payment",
                        "カード決済の引き落としが保留中になっている。",
                    ),
                    (
                        "refund_not_showing_up",
                        "返品・キャンセルの返金が口座に反映されない。",
                    ),
                    (
                        "transaction_charged_twice",
                        "1 回の買い物で同じ金額が二重に決済・請求された。",
                    ),
                    (
                        "unable_to_refund",
                        "アプリ上から返金処理・キャンセル処理を実行できない。",
                    ),
                ],
            ),
            (
                "cash_atm",
                "ATM 出金・現金引き出し・ATM 手数料トラブル",
                [
                    ("atm_support", "利用可能な提携 ATM や利用方法について知りたい。"),
                    (
                        "cash_withdrawal_charge",
                        "ATM 現金引き出し時に想定外の手数料が引かれた。",
                    ),
                    (
                        "cash_withdrawal_not_recognised",
                        "利用明細に身に覚えのない ATM 現金引き出しがある。",
                    ),
                    (
                        "declined_cash_withdrawal",
                        "ATM での現金引き出しが拒否・失敗した。",
                    ),
                    (
                        "supported_cards_currencies_at_atm",
                        "ATM で利用可能なカード種別や対応通貨を知りたい。",
                    ),
                    (
                        "wrong_amount_of_cash_received",
                        "ATM から指定と異なる金額の現金が出てきた。",
                    ),
                    (
                        "wrong_exchange_rate_for_cash_withdrawal",
                        "海外 ATM で出金した際の為替レートが不当に高い。",
                    ),
                    (
                        "contactless_not_working",
                        "非接触 (タッチ) 決済がリーダーで反応しない。",
                    ),
                    (
                        "declined_transfer",
                        "システム規制や不正検知により送金が強制拒否された。",
                    ),
                ],
            ),
            (
                "topup",
                "残高チャージ・入金反映・チャージ失敗および限度額",
                [
                    (
                        "automatic_top_up",
                        "残高連動のオートチャージ (自動入金) を設定・解除したい。",
                    ),
                    (
                        "balance_not_updated_after_cheque_or_cash_deposit",
                        "小切手や現金で入金したが口座残高に反映されない。",
                    ),
                    (
                        "pending_top_up",
                        "チャージ処理が処理中・保留のまま残高に加算されない。",
                    ),
                    (
                        "reverted_card_payment",
                        "カード決済が店舗側により取り消し・差し戻された。",
                    ),
                    (
                        "top_up_by_bank_transfer_charge",
                        "銀行振込による残高チャージで手数料が発生した。",
                    ),
                    (
                        "top_up_by_card_charge",
                        "クレジットカード等からのチャージで手数料が発生した。",
                    ),
                    (
                        "top_up_by_cash_or_cheque",
                        "現金や小切手による口座残高のチャージ方法を知りたい。",
                    ),
                    (
                        "top_up_failed",
                        "クレジットカード等からの残高チャージ操作が失敗した。",
                    ),
                    ("top_up_limits", "残高チャージの上限額や回数制限を確認したい。"),
                    (
                        "top_up_reverted",
                        "チャージした金額が取り消され元に戻ってしまった。",
                    ),
                    ("verify_top_up", "残高チャージ時に追加の認証や確認を求められた。"),
                ],
            ),
            (
                "account_sec",
                "口座設定・認証情報・紛失盗難対応・セキュリティ",
                [
                    ("change_pin", "カードの暗証番号 (PIN) を変更したい。"),
                    (
                        "compromised_card",
                        "カード番号が漏洩・不正利用された疑いがある。",
                    ),
                    (
                        "edit_personal_details",
                        "氏名、住所、電話番号などの登録個人情報を更新したい。",
                    ),
                    (
                        "lost_or_stolen_card",
                        "カードを紛失した、または盗難に遭ったため即時停止したい。",
                    ),
                    (
                        "lost_or_stolen_phone",
                        "アプリが登録されている携帯電話を紛失・盗難された。",
                    ),
                    (
                        "passcode_forgotten",
                        "アプリのログインパスコードやパスワードを忘れた。",
                    ),
                    (
                        "terminate_account",
                        "サービスを退会し、口座・アカウントを解約したい。",
                    ),
                    (
                        "verify_my_identity",
                        "口座開設や取引制限解除のための本人確認を行いたい。",
                    ),
                    (
                        "verify_source_of_funds",
                        "資金源や入金根拠に関する書類確認を求められた。",
                    ),
                    (
                        "why_verify_identity",
                        "なぜ本人確認書類の提出が必要なのか理由を知りたい。",
                    ),
                    (
                        "age_limit",
                        "サービスの利用年齢制限や未成年者の利用条件を知りたい。",
                    ),
                    (
                        "country_support",
                        "本サービスが利用可能な対応国・地域を確認したい。",
                    ),
                    (
                        "visa_or_mastercard",
                        "発行されるカードの国際ブランド (Visa/Mastercard) を知りたい。",
                    ),
                ],
            ),
            (
                "exchange_fee",
                "外貨両替・為替手数料・アプリ内両替および付帯サービス",
                [
                    (
                        "exchange_charge",
                        "通貨両替時に適用される手数料体系について知りたい。",
                    ),
                    (
                        "exchange_rate",
                        "現在の適用為替レートやリアルタイム換算レートを確認したい。",
                    ),
                    (
                        "exchange_via_app",
                        "アプリ内で複数の法定通貨・外貨を両替・保有したい。",
                    ),
                    (
                        "fiat_currency_support",
                        "サポートされている法定通貨の種類や制限を知りたい。",
                    ),
                    (
                        "pending_cash_withdrawal",
                        "ATM 現金引き出しの処理が完了待ちになっている。",
                    ),
                    (
                        "request_refund",
                        "店舗やサービス提供元に対して返金請求を行いたい。",
                    ),
                    (
                        "receiving_money",
                        "国内外からのお金の受け取り手順や受取口座情報を確認したい。",
                    ),
                    (
                        "transfer_fee_refund",
                        "誤って差し引かれた送金手数料の返還を求めたい。",
                    ),
                    (
                        "wrong_amount_refunded",
                        "店舗から返金された金額が購入金額と異なっている。",
                    ),
                    (
                        "other_or_unsupported",
                        "上記のいずれにも該当しないその他の問い合わせ。",
                    ),
                ],
            ),
        ]

        coarse_categories: dict[str, str] = {}
        fine_criteria: dict[str, str] = {}
        coarse_to_fine: dict[str, list[str]] = {}
        fine_to_coarse: dict[str, str] = {}
        all_coarse_keys: list[str] = []
        all_fine_keys: list[str] = []

        for coarse_key, coarse_desc, fine_items in categories:
            coarse_categories[coarse_key] = coarse_desc
            all_coarse_keys.append(coarse_key)
            coarse_to_fine[coarse_key] = []
            for fine_key, fine_desc in fine_items:
                fine_criteria[fine_key] = fine_desc
                coarse_to_fine[coarse_key].append(fine_key)
                fine_to_coarse[fine_key] = coarse_key
                all_fine_keys.append(fine_key)

        mapping = cls(
            name="Banking77-7Clusters",
            coarse_categories=coarse_categories,
            fine_criteria=fine_criteria,
            coarse_to_fine=coarse_to_fine,
            fine_to_coarse=fine_to_coarse,
            all_coarse_keys=all_coarse_keys,
            all_fine_keys=all_fine_keys,
            negative_keys=list(DEFAULT_NEGATIVE_KEYS),
        )
        mapping.validate()
        return mapping

    @classmethod
    def massive(cls) -> "HierarchicalMapping":
        """MASSIVE (ja-JP) 向け 18 シナリオ・60 インテントの階層プリセットを構築する。

        Amazon MASSIVE の階層構造 (scenario -> intent) に基づき、
        18 の粗分類シナリオと 60 の細分類インテントを相互排他的かつ網羅的に定義する。

        Returns:
            HierarchicalMapping: MASSIVE プリセットインスタンス。
        """
        categories: list[tuple[str, str, list[tuple[str, str]]]] = [
            (
                "alarm",
                "アラーム・目覚まし・タイマーの設定や確認・解除",
                [
                    ("alarm_set", "新しいアラームや目覚ましをセットする。"),
                    ("alarm_query", "設定されているアラーム時刻や状態を確認する。"),
                    ("alarm_remove", "設定済みのアラームを削除または解除する。"),
                ],
            ),
            (
                "audio",
                "音量調整・消音・オーディオ再生制御",
                [
                    ("audio_volume_mute", "音声をミュート (消音) にする。"),
                    ("audio_volume_up", "音量を上げる。"),
                    ("audio_volume_down", "音量を下げる。"),
                    ("audio_volume_other", "特定の音量レベルに変更する。"),
                ],
            ),
            (
                "calendar",
                "カレンダーの予定確認・新規予定作成・予定変更",
                [
                    ("calendar_set", "カレンダーに新しい予定を追加・登録する。"),
                    ("calendar_query", "登録されている予定やスケジュールを確認する。"),
                    ("calendar_remove", "予定を削除・キャンセルする。"),
                ],
            ),
            (
                "cooking",
                "レシピ検索・調理手順・計量や食材の質問",
                [
                    ("cooking_recipe", "料理のレシピや作り方・手順を検索する。"),
                    ("cooking_query", "調理法や代替食材・分量に関する質問をする。"),
                ],
            ),
            (
                "datetime",
                "現在日時・タイムゾーン・日付の計算や確認",
                [
                    ("datetime_query", "現在の日付や時刻・曜日を確認する。"),
                    ("datetime_convert", "異なるタイムゾーン間の時差を換算する。"),
                ],
            ),
            (
                "email",
                "メールの送受信・下書き作成・未読メール確認",
                [
                    ("email_sendemail", "指定した宛先へメールを送信する。"),
                    ("email_query", "受信メールや未読メールの内容を確認する。"),
                    ("email_querycontact", "アドレス帳や連絡先情報を検索する。"),
                    ("email_addcontact", "新しい連絡先をアドレス帳に追加する。"),
                ],
            ),
            (
                "general",
                "一般的な対話・挨拶・簡単な雑談応答",
                [
                    ("general_greet", "挨拶や声かけを行う。"),
                    ("general_joke", "ジョークや面白い話を聞く。"),
                    ("general_quirky", "雑談やユーモラスな質問をする。"),
                ],
            ),
            (
                "iot",
                "スマートホーム家電や照明・スマートプラグの操作",
                [
                    ("iot_hue_lighton", "照明やスマートライトを点灯する。"),
                    ("iot_hue_lightoff", "照明やスマートライトを消灯する。"),
                    ("iot_hue_lightdim", "照明の明るさを落とす (調光)。"),
                    ("iot_hue_lightup", "照明の明るさを上げる。"),
                    ("iot_hue_lightchange", "照明の色や発光パターンを変更する。"),
                    ("iot_cleaning", "ロボット掃除機の清掃を開始または停止する。"),
                    ("iot_wemo_on", "スマートプラグや家電の電源を入れる。"),
                    ("iot_wemo_off", "スマートプラグや家電の電源を切る。"),
                    ("iot_coffee", "スマートコーヒーメーカーで抽出を開始する。"),
                ],
            ),
            (
                "lists",
                "買い物リストやToDoリストのアイテム追加・削除・確認",
                [
                    ("lists_createoradd", "リストに新しいアイテムを追加・登録する。"),
                    ("lists_query", "リストの内容やアイテム一覧を確認する。"),
                    ("lists_remove", "リストからアイテムを削除する。"),
                ],
            ),
            (
                "music",
                "音楽の再生・一時停止・曲名検索やプレイリスト操作",
                [
                    ("music_likeness", "再生中の曲をお気に入り登録または高評価する。"),
                    (
                        "music_dislikeness",
                        "再生中の曲をスキップまたは低評価・嫌いに登録する。",
                    ),
                    ("music_query", "曲名やアーティスト情報・歌詞を検索する。"),
                    ("music_settings", "音楽再生のリピートやシャッフルを設定する。"),
                ],
            ),
            (
                "news",
                "最新ニュースやトピックスの読み上げ・検索",
                [
                    ("news_query", "最新のニュースや指定ジャンルの報道を確認する。"),
                ],
            ),
            (
                "play",
                "ゲーム・雑学クイズ・ポッドキャストやラジオ再生",
                [
                    ("play_music", "指定した楽曲・アルバム・プレイリストを再生する。"),
                    ("play_audiobook", "オーディオブックの朗読を再生する。"),
                    ("play_radio", "ラジオ放送やストリーミング配信を聴く。"),
                    ("play_game", "音声ゲームやクイズを開始する。"),
                    ("play_podcasts", "ポッドキャスト番組を再生する。"),
                ],
            ),
            (
                "qa",
                "事実確認・定義・人物や用語に関する一般的な質問回答",
                [
                    ("qa_factoid", "事実情報や定義・数値に関する質問をする。"),
                    ("qa_definition", "単語や用語の意味・定義を調べる。"),
                    ("qa_stock", "現在の株価や市場指標を確認する。"),
                    ("qa_currency", "為替相場や通貨換算レートを調べる。"),
                    ("qa_maths", "計算や単位換算の答えを求める。"),
                ],
            ),
            (
                "recommendation",
                "飲食店・映画・観光地などのおすすめ提案",
                [
                    ("recommendation_events", "開催予定のイベントや催し物を検索する。"),
                    (
                        "recommendation_locations",
                        "近隣の店舗や施設・おすすめスポットを探す。",
                    ),
                    ("recommendation_movies", "上映中の映画やおすすめ作品を探す。"),
                ],
            ),
            (
                "social",
                "SNS投稿・ソーシャルメッセージの送信や通知確認",
                [
                    ("social_post", "SNS やソーシャルメディアに近況を投稿する。"),
                    ("social_query", "SNS のタイムラインや通知を確認する。"),
                ],
            ),
            (
                "takeaway",
                "フードデリバリー・出前・テイクアウトの注文と配達状況",
                [
                    ("takeaway_order", "料理のデリバリーやテイクアウトを注文する。"),
                    ("takeaway_query", "注文した料理の配達状況や店舗情報を確認する。"),
                ],
            ),
            (
                "transport",
                "交通案内・タクシー配車・運行情報や乗り換え検索",
                [
                    ("transport_taxi", "タクシーの配車や事前予約を依頼する。"),
                    ("transport_traffic", "道路の渋滞状況や通行止め情報を調べる。"),
                    (
                        "transport_ticket",
                        "電車の乗車券や航空券の空席を確認・予約する。",
                    ),
                    ("transport_query", "公共交通機関の時刻表や乗り換え案内を調べる。"),
                ],
            ),
            (
                "weather",
                "現在の天気・週間天気予報・降水確率の確認",
                [
                    ("weather_query", "現在の天気や今後の天気予報・気温を確認する。"),
                ],
            ),
        ]

        coarse_categories: dict[str, str] = {}
        fine_criteria: dict[str, str] = {}
        coarse_to_fine: dict[str, list[str]] = {}
        fine_to_coarse: dict[str, str] = {}
        all_coarse_keys: list[str] = []
        all_fine_keys: list[str] = []

        for coarse_key, coarse_desc, fine_items in categories:
            coarse_categories[coarse_key] = coarse_desc
            all_coarse_keys.append(coarse_key)
            coarse_to_fine[coarse_key] = []
            for fine_key, fine_desc in fine_items:
                fine_criteria[fine_key] = fine_desc
                coarse_to_fine[coarse_key].append(fine_key)
                fine_to_coarse[fine_key] = coarse_key
                all_fine_keys.append(fine_key)

        mapping = cls(
            name="MASSIVE-ja-JP-18-60",
            coarse_categories=coarse_categories,
            fine_criteria=fine_criteria,
            coarse_to_fine=coarse_to_fine,
            fine_to_coarse=fine_to_coarse,
            all_coarse_keys=all_coarse_keys,
            all_fine_keys=all_fine_keys,
            negative_keys=list(DEFAULT_NEGATIVE_KEYS),
        )
        mapping.validate()
        return mapping
