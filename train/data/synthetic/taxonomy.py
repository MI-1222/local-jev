"""実務トリアージ・業務規約ドメイン階層およびシードサンプラーモジュール。

EC、通信、金融、SaaS、情シスの大分類・中分類・制約ノードを定義し、
バランスの取れた多様なシードタスク仕様を生成する。
"""

import random
from dataclasses import dataclass

from data.schema import QuestionType
from data.synthetic.config import PrimitiveRatioConfig


@dataclass(frozen=True)
class DomainNode:
    """業務ドメインの階層ノード。

    Attributes:
        domain (str): 大分類 (例: 'EC・小売', '金融・決済')。
        category (str): 中分類 (例: '請求・引き落とし', '障害対応')。
        constraint_node (str): 小分類・制約ノード (例: '二重請求', '保証期間超過')。
        description (str): シナリオ背景の説明。
    """

    domain: str
    category: str
    constraint_node: str
    description: str


@dataclass(frozen=True)
class SeedSpecification:
    """合成データ生成のためのシード仕様。

    Attributes:
        seed_id (str): シード固有ID。
        domain_node (DomainNode): 選択された業務ドメイン階層。
        question_type (QuestionType): 決定プリミティブ種別。
        num_options: Choice の場合は候補数、Score の場合は段階数、Noul の場合は 2。
        hard_negative_focus (str): ハードネガティブとして意図すべき誤認要因。
    """

    seed_id: str
    domain_node: DomainNode
    question_type: QuestionType
    num_options: int
    hard_negative_focus: str


DOMAIN_TAXONOMY: list[DomainNode] = [
    # EC・小売
    DomainNode(
        domain="EC・小売",
        category="返品・返金・交換",
        constraint_node="初期不良かつ購入後30日超",
        description="初期不良の申告であるが、購入後規約日数を経過しており通常返品が不可で有償修理・メーカー保証案内となる境界事例。",
    ),
    DomainNode(
        domain="EC・小売",
        category="配送トラブル",
        constraint_node="置き配指定での未着・誤配送",
        description="配送完了ステータスだが受取人が未受領であり、再配送・調査依頼・警察届出のいずれかを判定する事例。",
    ),
    DomainNode(
        domain="EC・小売",
        category="注文キャンセル",
        constraint_node="発送準備完了後のキャンセル申告",
        description="出荷ライン通過後のため注文取消不可、受取拒否または返品フローへの誘導を要する事例。",
    ),
    # 通信・ネットワーク
    DomainNode(
        domain="通信・ネットワーク",
        category="接続障害・速度低下",
        constraint_node="特定端末のみWiFi切断",
        description="回線事業者起因ではなく宅内LANルータまたは端末設定側の問題と切り分けるトリアージ事例。",
    ),
    DomainNode(
        domain="通信・ネットワーク",
        category="契約・SIM切替",
        constraint_node="eSIM再発行時のEID入力不一致",
        description="物理SIMとeSIMの混同、端末プロファイル未削除によるアクティベーション失敗の診断事例。",
    ),
    DomainNode(
        domain="通信・ネットワーク",
        category="海外ローミング",
        constraint_node="データローミング設定OFFによる不通",
        description="現地キャリア接続はできているが端末側の通信設定不備による遮断事例。",
    ),
    # 金融・決済
    DomainNode(
        domain="金融・決済",
        category="不正利用・請求疑義",
        constraint_node="海外加盟店からの身に覚えのない少額連続決済",
        description="即時カード一時停止と不正利用申告窓口へのエスカレーション判定事例。",
    ),
    DomainNode(
        domain="金融・決済",
        category="振込・口座引落",
        constraint_node="残高不足による再振替期限超過",
        description="自動再引き落とし不可、専用振込口座案内またはコンビニ納付書再発行の規約判定事例。",
    ),
    DomainNode(
        domain="金融・決済",
        category="本人確認・KYC",
        constraint_node="旧姓併記マイナンバーカードの厚み画像不鮮明",
        description="再撮影要請か有人書類確認トリアージかの自動判定事例。",
    ),
    # クラウド・SaaS
    DomainNode(
        domain="クラウド・SaaS",
        category="API・インフラ障害",
        constraint_node="504 Gateway Timeout かつ外部決済API依存",
        description="サーキットブレーカー発動、外部フェイルオーバー、クライアントリトライ制御の優先度決定事例。",
    ),
    DomainNode(
        domain="クラウド・SaaS",
        category="認証・SSO",
        constraint_node="SAMLアサーション有効期限切れによるログインループ",
        description="クライアント側クッキー破損ではなくIdP側クロックスキュー不一致の診断事例。",
    ),
    DomainNode(
        domain="クラウド・SaaS",
        category="権限管理・課金",
        constraint_node="管理者シート数超過によるメンバー追加拒否",
        description="プランアップグレード申請か、非アクティブユーザー整理かの推奨判定事例。",
    ),
    # 社内情シス・コーポレートIT
    DomainNode(
        domain="社内情シス・コーポレートIT",
        category="端末・セキュリティ",
        constraint_node="EDRによる未知スクリプトブロック検知",
        description="開発者の正当なスクリプトか、マルウェア隔離・端末ネットワーク遮断かを判断する事例。",
    ),
    DomainNode(
        domain="社内情シス・コーポレートIT",
        category="VPN・リモートアクセス",
        constraint_node="証明書失効後のリモートワーク接続試行",
        description="ヘルプデスク一時キー発行か、MDMプロファイル再配布かのトリアージ事例。",
    ),
    DomainNode(
        domain="社内情シス・コーポレートIT",
        category="アカウント・退職対応",
        constraint_node="退職者アカウントのGoogle Workspaceアーカイブとデータ委託",
        description="アカウント即時削除ではなくアーカイブライセンス移行・共有ドライブ移譲ルールの適用判定事例。",
    ),
]

HARD_NEGATIVE_FACTORS: list[str] = [
    "文脈のキーワードと完全一致するが、規約の例外規定（30日経過、有償契約限定等）により除外される候補。",
    "一見すると根本原因に見えるが、別の主管部署（ハードウェア、提携先、IdP）の責任範囲である候補。",
    "過去の暫定対処法としては正しかったが、現行ポリシー改訂により禁止されている古い運用手順の候補。",
    "緊急度が高そうに見えるが、実際には定常監視・経過観察で足りる過剰対応候補。",
    "一般的な推奨策だが、対象ユーザーの契約プラン（無料プラン・スタンダード）では提供されていない上位機能候補。",
]


class DomainTaxonomySampler:
    """業務ドメイン階層からシード仕様を公平にサンプリングするクラス。

    Attributes:
        config (PrimitiveRatioConfig): プリミティブ比率設定。
        rng (random.Random): 乱数生成器。
    """

    def __init__(
        self,
        config: PrimitiveRatioConfig | None = None,
        seed: int = 42,
    ) -> None:
        """サンプラーを初期化する。

        Args:
            config (PrimitiveRatioConfig | None): プリミティブ比率設定。None時はデフォルト。
            seed (int): 乱数シード。
        """
        self.config = config or PrimitiveRatioConfig()
        self.rng = random.Random(seed)

    def sample_question_type(self) -> QuestionType:
        """設定された比率に基づき決定プリミティブをサンプリングする。

        Returns:
            QuestionType: 決定プリミティブ。
        """
        weights = [
            self.config.choice_ratio,
            self.config.score_ratio,
            self.config.noul_ratio,
        ]
        types = [QuestionType.CHOICE, QuestionType.SCORE, QuestionType.NOUL]
        return self.rng.choices(types, weights=weights, k=1)[0]

    def sample_num_options(self, question_type: QuestionType) -> int:
        """決定プリミティブに応じた候補数・段階数をサンプリングする。

        Args:
            question_type (QuestionType): 決定プリミティブ。

        Returns:
            int: 候補数または段階数。
        """
        if question_type == QuestionType.NOUL:
            return 2

        if question_type == QuestionType.SCORE:
            weights = [self.config.score_3_5_ratio, self.config.score_10_ratio]
            category_val = self.rng.choices(["3_5", "10"], weights=weights, k=1)[0]
            if category_val == "3_5":
                return self.rng.choice([3, 4, 5])
            return 10

        weights = [
            self.config.choice_2_3_ratio,
            self.config.choice_4_6_ratio,
            self.config.choice_7_16_ratio,
        ]
        bucket_val = self.rng.choices(["2_3", "4_6", "7_16"], weights=weights, k=1)[0]
        if bucket_val == "2_3":
            return self.rng.choice([2, 3])
        if bucket_val == "4_6":
            return self.rng.choice([4, 5, 6])
        return self.rng.choice(list(range(7, 17)))

    def generate_seed_spec(self, sequence_index: int) -> SeedSpecification:
        """シード仕様インスタンスを生成する。

        Args:
            sequence_index (int): 連番インデックス。

        Returns:
            SeedSpecification: 構築されたシード仕様。
        """
        domain_node = self.rng.choice(DOMAIN_TAXONOMY)
        q_type = self.sample_question_type()
        num_opts = self.sample_num_options(q_type)
        factor = self.rng.choice(HARD_NEGATIVE_FACTORS)
        seed_id = f"synth_{q_type.value}_{sequence_index:06d}"

        return SeedSpecification(
            seed_id=seed_id,
            domain_node=domain_node,
            question_type=q_type,
            num_options=num_opts,
            hard_negative_focus=factor,
        )
