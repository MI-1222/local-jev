"""MASSIVE (ja-JP) 多言語仮想アシスタントデータセットコンバータモジュール。

Amazon MASSIVE データセット (ja-JP ロケール) を、Jev 統一スキーマ (Choice 型)
へ正規化・変換する。18 シナリオ (大分類) と 60 インテント (細分類) の
階層構造を保持し、粗密階層ルーティングのベンチマークおよび学習データとして活用する。
"""

import random
from collections.abc import Iterator

from datasets import Dataset, load_dataset

from data.converters.base import BaseDatasetConverter
from data.hierarchical import HierarchicalMapping
from data.prompt_pool import sample_instruction
from data.schema import QuestionType, UnifiedSample

# MASSIVE (ja-JP) 18 シナリオの日本語説明文マッピング
MASSIVE_SCENARIO_DESCRIPTIONS: dict[str, str] = {
    "alarm": "アラーム・目覚まし・タイマーの設定や確認・解除",
    "audio": "音量調整・消音・オーディオ再生制御",
    "calendar": "カレンダーの予定確認・新規予定作成・予定変更",
    "cooking": "レシピ検索・調理手順・計量や食材の質問",
    "datetime": "現在日時・タイムゾーン・日付の計算や確認",
    "email": "メールの送受信・下書き作成・未読メール確認",
    "general": "一般的な対話・挨拶・簡単な雑談応答",
    "iot": "スマートホーム家電や照明・スマートプラグの操作",
    "lists": "買い物リストやToDoリストのアイテム追加・削除・確認",
    "music": "音楽の再生・一時停止・曲名検索やプレイリスト操作",
    "news": "最新ニュースやトピックスの読み上げ・検索",
    "play": "ゲーム・雑学クイズ・ポッドキャストやラジオ再生",
    "qa": "事実確認・定義・人物や用語に関する一般的な質問回答",
    "recommendation": "飲食店・映画・観光地などのおすすめ提案",
    "social": "SNS投稿・ソーシャルメッセージの送信や通知確認",
    "takeaway": "フードデリバリー・出前・テイクアウトの注文と配達状況",
    "transport": "交通案内・タクシー配車・運行情報や乗り換え検索",
    "weather": "現在の天気・週間天気予報・降水確率の確認",
}

# MASSIVE 60 インテントのシナリオ別マッピングおよび日本語説明文
MASSIVE_INTENT_DEFINITIONS: dict[str, dict[str, str]] = {
    "alarm": {
        "alarm_set": "新しいアラームや目覚ましをセットする。",
        "alarm_query": "設定されているアラーム時刻や状態を確認する。",
        "alarm_remove": "設定済みのアラームを削除または解除する。",
    },
    "audio": {
        "audio_volume_mute": "音声をミュート (消音) にする。",
        "audio_volume_up": "音量を上げる。",
        "audio_volume_down": "音量を下げる。",
        "audio_volume_other": "特定の音量レベルに変更する。",
    },
    "calendar": {
        "calendar_set": "カレンダーに新しい予定を追加・登録する。",
        "calendar_query": "登録されている予定やスケジュールを確認する。",
        "calendar_remove": "予定を削除・キャンセルする。",
    },
    "cooking": {
        "cooking_recipe": "料理のレシピや作り方・手順を検索する。",
        "cooking_query": "調理法や代替食材・分量に関する質問をする。",
    },
    "datetime": {
        "datetime_query": "現在の日付や時刻・曜日を確認する。",
        "datetime_convert": "異なるタイムゾーン間の時差を換算する。",
    },
    "email": {
        "email_sendemail": "指定した宛先へメールを送信する。",
        "email_query": "受信メールや未読メールの内容を確認する。",
        "email_querycontact": "アドレス帳や連絡先情報を検索する。",
        "email_addcontact": "新しい連絡先をアドレス帳に追加する。",
    },
    "general": {
        "general_greet": "挨拶や声かけを行う。",
        "general_joke": "ジョークや面白い話を聞く。",
        "general_quirky": "雑談やユーモラスな質問をする。",
    },
    "iot": {
        "iot_hue_lighton": "照明やスマートライトを点灯する。",
        "iot_hue_lightoff": "照明やスマートライトを消灯する。",
        "iot_hue_lightdim": "照明の明るさを落とす (調光)。",
        "iot_hue_lightup": "照明の明るさを上げる。",
        "iot_hue_lightchange": "照明の色や発光パターンを変更する。",
        "iot_cleaning": "ロボット掃除機の清掃を開始または停止する。",
        "iot_wemo_on": "スマートプラグや家電の電源を入れる。",
        "iot_wemo_off": "スマートプラグや家電の電源を切る。",
        "iot_coffee": "スマートコーヒーメーカーで抽出を開始する。",
    },
    "lists": {
        "lists_createoradd": "リストに新しいアイテムを追加・登録する。",
        "lists_query": "リストの内容やアイテム一覧を確認する。",
        "lists_remove": "リストからアイテムを削除する。",
    },
    "music": {
        "music_likeness": "再生中の曲をお気に入り登録または高評価する。",
        "music_dislikeness": "再生中の曲をスキップまたは低評価・嫌いに登録する。",
        "music_query": "曲名やアーティスト情報・歌詞を検索する。",
        "music_settings": "音楽再生のリピートやシャッフルを設定する。",
    },
    "news": {
        "news_query": "最新のニュースや指定ジャンルの報道を確認する。",
    },
    "play": {
        "play_music": "指定した楽曲・アルバム・プレイリストを再生する。",
        "play_audiobook": "オーディオブックの朗読を再生する。",
        "play_radio": "ラジオ放送やストリーミング配信を聴く。",
        "play_game": "音声ゲームやクイズを開始する。",
        "play_podcasts": "ポッドキャスト番組を再生する。",
    },
    "qa": {
        "qa_factoid": "事実情報や定義・数値に関する質問をする。",
        "qa_definition": "単語や用語の意味・定義を調べる。",
        "qa_stock": "現在の株価や市場指標を確認する。",
        "qa_currency": "為替相場や通貨換算レートを調べる。",
        "qa_maths": "計算や単位換算の答えを求める。",
    },
    "recommendation": {
        "recommendation_events": "開催予定のイベントや催し物を検索する。",
        "recommendation_locations": "近隣の店舗や施設・おすすめスポットを探す。",
        "recommendation_movies": "上映中の映画やおすすめ作品を探す。",
    },
    "social": {
        "social_post": "SNS やソーシャルメディアに近況を投稿する。",
        "social_query": "SNS のタイムラインや通知を確認する。",
    },
    "takeaway": {
        "takeaway_order": "料理のデリバリーやテイクアウトを注文する。",
        "takeaway_query": "注文した料理の配達状況や店舗情報を確認する。",
    },
    "transport": {
        "transport_taxi": "タクシーの配車や事前予約を依頼する。",
        "transport_traffic": "道路の渋滞状況や通行止め情報を調べる。",
        "transport_ticket": "電車の乗車券や航空券の空席を確認・予約する。",
        "transport_query": "公共交通機関の時刻表や乗り換え案内を調べる。",
    },
    "weather": {
        "weather_query": "現在の天気や今後の天気予報・気温を確認する。",
    },
}


def build_massive_hierarchical_mapping() -> HierarchicalMapping:
    """MASSIVE (ja-JP) 向け 18 シナリオ・60 インテントの階層マッピングを構築する。

    Returns:
        HierarchicalMapping: MASSIVE 向け階層オントロジーマッピング。
    """
    return HierarchicalMapping.massive()


class MASSIVEConverter(BaseDatasetConverter):
    """MASSIVE (ja-JP) データセットコンバータ。

    Attributes:
        hierarchical_mapping (HierarchicalMapping): MASSIVE 階層オントロジーマッピング。
        max_negative_options (int | None): 訓練時サブサンプリングする細分類負例候補数。
    """

    def __init__(
        self,
        max_negative_options: int | None = 7,
        seed: int = 42,
    ) -> None:
        """MASSIVE コンバータを初期化する。

        Args:
            max_negative_options (int | None): 訓練時サブサンプリングする負例数。
            seed (int): 乱数シード。
        """
        super().__init__(seed=seed)
        self.max_negative_options = max_negative_options
        self.hierarchical_mapping = build_massive_hierarchical_mapping()

    def convert_dataset(
        self,
        dataset: Dataset,
        split: str,
    ) -> Iterator[UnifiedSample]:
        """与えられた Hugging Face Dataset を走査して UnifiedSample を生成する。

        Args:
            dataset (Dataset): 対象の Dataset オブジェクト。
            split (str): スプリット名 (train, validation, test)。

        Yields:
            Iterator[UnifiedSample]: 変換後の統一サンプル列。
        """
        rng = random.Random(self.seed)
        all_fine_keys = self.hierarchical_mapping.all_fine_keys
        fine_criteria = self.hierarchical_mapping.fine_criteria
        fine_to_coarse = self.hierarchical_mapping.fine_to_coarse

        for idx, row in enumerate(dataset):
            target_intent = str(row.get("intent", ""))
            if not target_intent or target_intent not in fine_criteria:
                continue

            text = str(row.get("utt", ""))
            scenario = row.get("scenario") or fine_to_coarse.get(target_intent, "")

            # 訓練時かつ max_negative_options が指定されている場合は負例をサンプリング
            if (
                split == "train"
                and self.max_negative_options is not None
                and len(all_fine_keys) > self.max_negative_options + 1
            ):
                negatives = [k for k in all_fine_keys if k != target_intent]
                selected_negs = rng.sample(negatives, self.max_negative_options)
                candidate_ids = [target_intent] + selected_negs
            else:
                candidate_ids = list(all_fine_keys)

            if target_intent not in candidate_ids:
                candidate_ids.append(target_intent)

            criteria = {cid: fine_criteria[cid] for cid in candidate_ids}
            instructions = sample_instruction("intent", rng=rng)

            yield UnifiedSample(
                dataset_name="massive_ja",
                sample_id=f"massive_ja_{split}_{idx}",
                question_type=QuestionType.CHOICE,
                state=text,
                instructions=instructions,
                criteria=criteria,
                target=target_intent,
                metadata={
                    "split": split,
                    "scenario": scenario,
                    "num_options": len(criteria),
                },
            )

    def convert_split(self, split: str) -> Iterator[UnifiedSample]:
        """Hugging Face から MASSIVE (ja-JP) をロードして変換する。

        Args:
            split (str): スプリット名 (train, validation, test)。

        Yields:
            Iterator[UnifiedSample]: 統一サンプル列。
        """
        hf_split = "validation" if split in ["val", "validation"] else split
        try:
            ds = load_dataset(
                "qanastek/MASSIVE",
                "ja-JP",
                split=hf_split,
            )
        except (RuntimeError, ValueError, OSError):
            ds = load_dataset(
                "AmazonScience/massive",
                "ja-JP",
                split=hf_split,
            )
        assert isinstance(ds, Dataset)
        yield from self.convert_dataset(ds, split=split)
