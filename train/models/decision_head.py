"""動的スキーマ対応デシジョンヘッドおよび Jev 統合モデルモジュール。

バックボーンエンコーダの最終隠れ状態から、各候補マーカー `[OP]` の隠れベクトルを
Gather 抽出層で抽出し、置換同変アテンション (Set Attention Block: SAB)、
およびタスク特性に応じた数理射影ヘッド (Choice / CORAL 順序回帰 / NLI 潜在表現) を経由して
型安全・高精度な決定ロジットおよび確率分布を算出する。
"""

import logging
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel

from contract import (
    TENSOR_ATTENTION_MASK,
    TENSOR_INPUT_IDS,
    TENSOR_LOGITS,
    TENSOR_OP_INDICES,
)
from data.schema import QuestionType

logger = logging.getLogger(__name__)

DEFAULT_MASK_VALUE = -1e4
"""無効候補に適用する負の無限大代替値。半精度アンダーフローや NaN を防止する。"""


class OptionGatherLayer(nn.Module):
    """`[OP]` トークン位置の隠れベクトルを抽出する Gather 層。

    バックボーンが出力する全トークンの隠れ状態系列から、
    候補位置インデックスに対応するスライスを並列に収集する。
    無効候補 (-1) が含まれる場合でも安全にクランプして抽出し、
    無効位置をゼロベクトルでマスクする。
    """

    def forward(self, hidden_states: Tensor, op_indices: Tensor) -> Tensor:
        """隠れ状態テンソルから op_indices で指定された位置のベクトルを抽出する。

        内部ロジック:
        1. 隠れ層の次元数 `hidden_size` を取得する。
        2. `op_indices` を 0 以上にクランプして安全なインデックス列を生成する。
        3. `torch.gather` を系列長次元 (`dim=1`) に適用してマーカーベクトルを抽出する。
        4. 元の `op_indices` が -1 の位置をゼロベクトルでマスクする。

        Args:
            hidden_states (Tensor): バックボーン最終層の隠れ状態 `[batch_size, seq_len, hidden_size]`。
            op_indices (Tensor): 各候補の `[OP]` トークン位置インデックス `[batch_size, num_options]`。

        Returns:
            Tensor: 候補マーカー隠れベクトル `[batch_size, num_options, hidden_size]`。
        """
        hidden_size = hidden_states.shape[-1]
        safe_indices = op_indices.clamp(min=0)
        expanded_indices = safe_indices.unsqueeze(-1).expand(-1, -1, hidden_size)
        gathered = torch.gather(hidden_states, dim=1, index=expanded_indices)
        valid_mask = (op_indices != -1).unsqueeze(-1).to(gathered.dtype)
        return gathered * valid_mask


class SetAttentionBlock(nn.Module):
    """位置エンコーディングを持たない置換同変アテンションブロック (Set Attention Block: SAB)。

    候補マーカー表現間の相互比較を可能にしつつ、提示順序に対する置換同変性
    (Permutation Equivariance) を数学的に保証する。
    ONNX エクスポート互換性を最大化するため、未サポートカーネルを回避し
    標準的なテンソル演算 (MatMul, Softmax, Transpose) で自己注意を構成する。

    数理仕様:
    位置埋め込みを含めない自己注意機構と残差接続・LayerNorm により構成される：
    $$H_{\\text{sab}} = \\text{LayerNorm}(H_{\\text{op}} + \\text{MHA}(Q=H_{\\text{op}}, K=H_{\\text{op}}, V=H_{\\text{op}}))$$
    任意の置換行列 $\\Pi$ に対し、$\\text{SAB}(\\Pi H) = \\Pi \\text{SAB}(H)$ が厳密に成立する。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        """SAB ブロックを初期化する。

        Args:
            hidden_size (int): 隠れ層次元数。
            num_heads (int): マルチヘッドアテンションのヘッド数。
            dropout (float): ドロップアウト率。
        """
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) は num_heads ({num_heads}) で割り切れる必要があります。"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        option_vectors: Tensor,
        op_mask: Tensor | None = None,
    ) -> Tensor:
        """候補マーカーベクトルに対して置換同変自己注意を適用する。

        内部ロジック:
        1. Q, K, V を線形変換し、[B, num_heads, K, head_dim] に転置する。
        2. Scaled Dot-Product Attention を計算する: scores = Q K^T / sqrt(d_k)。
        3. `op_mask` が提供されている場合、無効候補位置の注意重みを -1e4 で遮断する。
        4. Softmax 正規化および V との内積により集約表現を得る。
        5. 出力線形変換、残差加算、および LayerNorm 正規化を施す。

        Args:
            option_vectors (Tensor): 候補マーカーベクトル `[batch_size, num_options, hidden_size]`。
            op_mask (Tensor | None): 有効候補マスク `[batch_size, num_options]` (有効: True, 無効: False)。

        Returns:
            Tensor: 置換同変特徴量テンソル `[batch_size, num_options, hidden_size]`。
        """
        b_sz, num_opts, _ = option_vectors.shape

        # Q, K, V の線形射影とマルチヘッド展開: [B, num_heads, K, head_dim]
        q = (
            self.q_proj(option_vectors)
            .view(b_sz, num_opts, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(option_vectors)
            .view(b_sz, num_opts, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(option_vectors)
            .view(b_sz, num_opts, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Scaled Dot-Product: [B, num_heads, K, K]
        scale = float(self.head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if op_mask is not None:
            # key_padding_mask: [B, 1, 1, K] で Key 次元にブロードキャスト
            key_mask = (~op_mask).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_mask, DEFAULT_MASK_VALUE)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [B, num_heads, K, head_dim] -> [B, K, hidden_size]
        attn_out = torch.matmul(attn_weights, v)
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(b_sz, num_opts, self.hidden_size)
        )
        attn_out = self.out_proj(attn_out)

        normed = self.layer_norm(option_vectors + attn_out)

        if op_mask is not None:
            normed = normed * op_mask.unsqueeze(-1).to(normed.dtype)

        return normed


class ChoiceHead(nn.Module):
    """2層 MLP による Choice プリミティブ向け候補ロジット射影ヘッド。

    数理仕様:
    $$z_i = W_2 \\cdot \\text{GELU}(W_1 h_i + b_1) + b_2$$
    無効候補スロットに対しては負のマスク値 (デフォルト: -1e4) を付与する。
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden_size: int | None = None,
        dropout: float = 0.0,
        mask_value: float = DEFAULT_MASK_VALUE,
    ) -> None:
        """Choice ヘッドを初期化する。

        Args:
            hidden_size (int): バックボーンの隠れ層次元数。
            mlp_hidden_size (int | None): MLP 中間層次元数 (未指定時は `hidden_size // 2`)。
            dropout (float): ドロップアウト率。
            mask_value (float): 無効候補に代入する負のマスク値。
        """
        super().__init__()
        mid_size = mlp_hidden_size if mlp_hidden_size is not None else hidden_size // 2
        self.dense = nn.Linear(hidden_size, mid_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.out_proj = nn.Linear(mid_size, 1)
        self.mask_value = mask_value

    def forward(
        self,
        option_vectors: Tensor,
        op_mask: Tensor | None = None,
    ) -> Tensor:
        """候補マーカーベクトルから候補ごとのスカラーロジットを算出する。

        Args:
            option_vectors (Tensor): 候補マーカーベクトル `[batch_size, num_options, hidden_size]`。
            op_mask (Tensor | None): 有効候補マスク `[batch_size, num_options]`。

        Returns:
            Tensor: 候補ロジットテンソル `[batch_size, num_options]`。
        """
        hidden = self.dropout(self.act(self.dense(option_vectors)))
        logits = self.out_proj(hidden).squeeze(-1)
        if op_mask is not None:
            logits = logits.masked_fill(~op_mask, self.mask_value)
        return logits


DecisionHead = ChoiceHead
"""後方互換性のためのエイリアス。"""


class CoralOrdinalHead(nn.Module):
    """CORAL (Consistent Rank Logits) 累積リンク順序回帰ヘッド。

    順序尺度 (Score: 2〜10 段階) において多クラス Softmax を排し、
    M-1 個の累積確率 $P(Y > k) = \\sigma(w^T h - b_k)$ を算出する。
    単調増加バイアス $b_1 \\le b_2 \\le \\dots \\le b_{M-1}$ の再パラメータ化により、
    最適化のいかなる過程でも累積確率の単調減少性およびクラス確率の非負性を 100% 保証する。

    数理仕様:
    単調増加バイアスの生成:
    $$b_1 = \\theta_1, \\quad b_k = b_1 + \\sum_{j=2}^k \\text{softplus}(\\theta_j)$$
    累積確率:
    $$P(Y > k) = \\sigma(w^T h - b_k) \\quad (k = 1, \\dots, M-1)$$
    各段階の確率質量:
    $$P(Y = 1) = 1 - P(Y > 1)$$
    $$P(Y = k) = P(Y > k-1) - P(Y > k) \\quad (2 \\le k \\le M-1)$$
    $$P(Y = M) = P(Y > M-1)$$
    """

    def __init__(
        self,
        hidden_size: int,
        max_levels: int = 10,
    ) -> None:
        """CORAL 順序回帰ヘッドを初期化する。

        Args:
            hidden_size (int): 隠れ層次元数。
            max_levels (int): サポートする最大評価段階数 (デフォルト: 10)。
        """
        super().__init__()
        self.max_levels = max_levels
        self.feature_proj = nn.Linear(hidden_size, 1, bias=False)

        # 単調バイアスパラメータ: 初項は -1.0、増分パラメータは 0.0 付近で初期化
        init_biases = torch.zeros(max_levels - 1)
        init_biases[0] = -1.0
        self.raw_biases = nn.Parameter(init_biases)

    def get_ordered_biases(self) -> Tensor:
        """単調増加バイアス列 $b_1 \\le b_2 \\le \\dots \\le b_{M-1}$ を生成する。

        Returns:
            Tensor: 単調増加バイアステンソル `[max_levels - 1]`。
        """
        first_bias = self.raw_biases[0:1]
        increments = F.softplus(self.raw_biases[1:])
        return torch.cumsum(torch.cat([first_bias, increments], dim=0), dim=0)

    def forward(
        self,
        aggregated_vector: Tensor,
        num_levels: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """集約文脈表現から CORAL 順序確率分布およびロジットを算出する。

        内部ロジック:
        1. 共有重みにより特徴スカラー $w^T h$ を算出する。
        2. $num_levels - 1$ 個の単調増加バイアスをスライスして累積ロジットを計算する。
        3. Sigmoid により単調減少な累積確率列 $P(Y > k)$ を算出する。
        4. 隣接差分から各段階の確率質量 $P(Y = k)$ を導出し、クランプと正規化を行う。
        5. Rust/ONNX ランタイム契約互換の対数確率ロジット $\\ln(p_k)$ を逆算する。

        Args:
            aggregated_vector (Tensor): 文脈または候補集約表現 `[batch_size, hidden_size]`。
            num_levels (int): 評価段階数 $M \\in [2, max\\_levels]$。

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                - probs: 各段階の確率質量分布 `[batch_size, num_levels]`。
                - logits: Rust/ONNX 契約互換のロジット `[batch_size, num_levels]`。
                - cum_probs: 累積確率列 `[batch_size, num_levels - 1]`。
        """
        feat = self.feature_proj(aggregated_vector)  # [B, 1]
        ordered_biases = self.get_ordered_biases()[: num_levels - 1]  # [M - 1]

        # P(Y > k) = sigmoid(feat - b_k). b_k が単調増加するため、feat - b_k は単調減少し、P(Y > k) も単調減少する
        cum_logits = feat - ordered_biases.unsqueeze(0)  # [B, M - 1]
        cum_probs = torch.sigmoid(cum_logits)  # [B, M - 1]

        p_first = 1.0 - cum_probs[:, 0:1]
        p_middle = cum_probs[:, :-1] - cum_probs[:, 1:]
        p_last = cum_probs[:, -1:]

        probs = torch.cat([p_first, p_middle, p_last], dim=-1)
        probs = probs.clamp(min=1e-8, max=1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Softmax(ln(probs)) = probs となるため、Rust 側の Softmax 処理と完全に整合する
        logits = torch.log(probs)

        return probs, logits, cum_probs


class NliNoulHead(nn.Module):
    """NLI (自然言語推論: 含意・中立・矛盾) 潜在表現を統合した Noul ヘッド。

    指示文と文脈の論理的関係性を 3 クラス (含意 $z_e$、中立 $z_n$、矛盾 $z_c$) のロジットとして抽出し、
    中立 (情報不足・曖昧さ) を分離した条件付き真実確率 $P(\\text{True}) = \\sigma(z_e - z_c)$、
    および中立不確実性 $P(\\text{Uncertain}) = p_n$ を導出する。
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden_size: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        """NLI Noul ヘッドを初期化する。

        Args:
            hidden_size (int): 隠れ層次元数。
            mlp_hidden_size (int | None): MLP 中間層次元数 (未指定時は `hidden_size // 2`)。
            dropout (float): ドロップアウト率。
        """
        super().__init__()
        mid_size = mlp_hidden_size if mlp_hidden_size is not None else hidden_size // 2
        self.dense = nn.Linear(hidden_size, mid_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.out_proj = nn.Linear(mid_size, 3)

    def forward(
        self,
        feature_vector: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """3 クラス NLI ロジットから真偽判定結果および不確実性を算出する。

        内部ロジック:
        1. 2層 MLP により 3 クラスロジット $[z_e, z_n, z_c]$ を算出する。
        2. Softmax により 3 クラス確率 $[p_e, p_n, p_c]$ を得る。
        3. 条件付き真実確率 $P(\\text{True}) = \\frac{p_e}{p_e + p_c} = \\sigma(z_e - z_c)$ を導出する。
        4. 中立確率 $p_n$ を情報不足不確実性として保持する。
        5. Rust ランタイム契約互換の 2 クラスロジット $[z_e, z_c]$ を出力する。

        Args:
            feature_vector (Tensor): Noul 特徴量ベクトル `[batch_size, hidden_size]`。

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]:
                - logits_2class: Rust/ONNX 契約互換の 2 候補ロジット `[batch_size, 2]`。
                - p_true: 条件付き真実確率 `[batch_size]`。
                - uncertainty: 中立不確実性 `[batch_size]`。
                - logits_3class: 生の 3 クラスロジット `[batch_size, 3]`。
        """
        hidden = self.dropout(self.act(self.dense(feature_vector)))
        logits_3class = self.out_proj(hidden)  # [B, 3]

        z_e = logits_3class[:, 0]
        z_c = logits_3class[:, 2]

        probs_3class = F.softmax(logits_3class, dim=-1)
        p_e = probs_3class[:, 0]
        p_n = probs_3class[:, 1]
        p_c = probs_3class[:, 2]

        # 条件付き真実確率: P(True) = p_e / (p_e + p_c) = sigma(z_e - z_c)
        p_true = p_e / (p_e + p_c).clamp(min=1e-8)
        uncertainty = p_n

        # Rust ランタイム側は delta_z = logits[0] - logits[1] でシグモイド計算するため、
        # [z_e, z_c] を渡すことで delta_z = z_e - z_c が厳密に再現される
        logits_2class = torch.stack([z_e, z_c], dim=-1)

        return logits_2class, p_true, uncertainty, logits_3class


class JevDecisionModel(nn.Module):
    """Jev アーキテクチャに準拠した非自己回帰型決定モデル (Phase 4 数理刷新版)。

    入力トークン列を受け取り、単一フォワードパスで各候補の決定ロジットを出力する。
    OptionGatherLayer -> SetAttentionBlock (SAB) -> タスク別射影ヘッドのパイプラインにより、
    位置バイアスの解消、順序尺度の幾何構造保持、および論理的真偽判定を実現する。
    ONNX エクスポートおよび推論ランタイムとの契約インターフェースに完全準拠する。
    """

    def __init__(
        self,
        backbone: PreTrainedModel,
        mlp_hidden_size: int | None = None,
        num_attention_heads: int = 4,
        sab_dropout: float = 0.0,
        head_dropout: float = 0.0,
        max_ordinal_levels: int = 10,
    ) -> None:
        """Jev 決定モデルを初期化する。

        Args:
            backbone (PreTrainedModel): 事前学習済みエンコーダバックボーン (ModernBERT / mmBERT 等)。
            mlp_hidden_size (int | None): デシジョンヘッドの MLP 中間次元数。
            num_attention_heads (int): SAB ブロックのアテンションヘッド数。
            sab_dropout (float): SAB ブロックのドロップアウト率。
            head_dropout (float): 各射影ヘッドのドロップアウト率。
            max_ordinal_levels (int): Score 型の最大サポート段階数。
        """
        super().__init__()
        self.backbone = backbone
        hidden_size = cast(int, backbone.config.hidden_size)

        self.gather_layer = OptionGatherLayer()
        self.sab = SetAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            dropout=sab_dropout,
        )

        self.choice_head = ChoiceHead(
            hidden_size=hidden_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=head_dropout,
        )
        self.score_head = CoralOrdinalHead(
            hidden_size=hidden_size,
            max_levels=max_ordinal_levels,
        )
        self.noul_head = NliNoulHead(
            hidden_size=hidden_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=head_dropout,
        )

        # 既存コードとの後方互換性プロパティ
        self.decision_head = self.choice_head

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        op_indices: Tensor,
        op_mask: Tensor | None = None,
        question_type: str | QuestionType | None = None,
        return_features: bool = False,
        return_details: bool = False,
    ) -> Tensor | tuple[Any, ...]:
        """単一フォワードパスで決定ロジットを算出する。

        内部ロジック:
        1. バックボーンエンコーダで全系列の隠れ状態 `last_hidden_state` を取得する。
        2. `OptionGatherLayer` で `op_indices` に対応する各候補マーカーの隠れベクトルを抽出する。
        3. `op_mask` が未指定の場合、`op_indices != -1` から有効候補マスクを自動導出する。
        4. `SetAttentionBlock` (SAB) により候補マーカー表現間の置換同変自己注意を適用する。
        5. `question_type` に応じて適切な射影ヘッド (Choice, CORAL Score, NLI Noul) を実行する。
        6. ONNX エクスポート時および通常推論時は契約仕様に従い `logits [batch_size, num_options]` を返却する。
        7. `return_features=True` または `return_details=True` の場合は中間状態を同時に返却する。

        Args:
            input_ids (Tensor): トークンID列 `[batch_size, seq_len]`。
            attention_mask (Tensor): アテンションマスク `[batch_size, seq_len]`。
            op_indices (Tensor): 候補マーカー位置インデックス `[batch_size, num_options]`。
            op_mask (Tensor | None): 有効候補マスク `[batch_size, num_options]`。
            question_type (str | QuestionType | None): 質問種別 (`choice`, `score`, `noul`)。
            return_features (bool): 文脈埋め込みおよび候補隠れ状態を同時に返却するかどうか。
            return_details (bool): 各ヘッドの詳細計算結果 (確率、不確実性等) を辞書で同時に返却するかどうか。

        Returns:
            Tensor | tuple[Any, ...]:
                - 通常時: 各候補のロジット `[batch_size, num_options]`。
                - 特徴量返却時: (logits, state_repr, option_vectors) のタプル。
                - 詳細返却時: (logits, state_repr, option_vectors, details_dict) のタプル。
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state: Tensor = outputs.last_hidden_state

        if op_mask is None:
            op_mask = op_indices != -1

        raw_option_vectors = self.gather_layer(last_hidden_state, op_indices)
        sab_option_vectors = self.sab(raw_option_vectors, op_mask=op_mask)

        # 文脈表現の平均プーリング
        mask_expanded = attention_mask.unsqueeze(-1).float()
        state_repr = (last_hidden_state * mask_expanded).sum(dim=1) / mask_expanded.sum(
            dim=1
        ).clamp(min=1e-8)

        # タスク種別の判定
        q_type_str: str | None = None
        if question_type is not None:
            q_type_str = (
                question_type.value
                if isinstance(question_type, QuestionType)
                else str(question_type).lower()
            )

        details: dict[str, Any] = {}

        if q_type_str == QuestionType.SCORE.value or q_type_str == "score":
            num_levels = op_indices.size(1)
            # 候補の平均表現を集約特徴量とする (有効候補のみで平均)
            valid_counts = op_mask.sum(dim=1, keepdim=True).clamp(min=1)
            agg_vector = (
                sab_option_vectors * op_mask.unsqueeze(-1).to(sab_option_vectors.dtype)
            ).sum(dim=1) / valid_counts

            probs, logits, cum_probs = self.score_head(agg_vector, num_levels)
            details["score_probs"] = probs
            details["cum_probs"] = cum_probs
        elif q_type_str == QuestionType.NOUL.value or q_type_str == "noul":
            # Noul では候補ベクトルの先頭差分または集約表現を使用
            if sab_option_vectors.size(1) >= 2:
                noul_feat = sab_option_vectors[:, 0] - sab_option_vectors[:, 1]
            else:
                noul_feat = state_repr

            logits_2class, p_true, uncertainty, logits_3class = self.noul_head(
                noul_feat
            )
            logits = logits_2class
            details["noul_p_true"] = p_true
            details["noul_uncertainty"] = uncertainty
            details["noul_logits_3class"] = logits_3class
        else:
            # デフォルトは Choice
            logits = self.choice_head(sab_option_vectors, op_mask=op_mask)

        if return_details:
            return logits, state_repr, sab_option_vectors, details

        if return_features:
            return logits, state_repr, sab_option_vectors

        return logits

    @classmethod
    def get_input_names(cls) -> list[str]:
        """ONNX エクスポート時の入力テンソル名リストを取得する。

        Returns:
            list[str]: 入力テンソル名のリスト。
        """
        return [TENSOR_INPUT_IDS, TENSOR_ATTENTION_MASK, TENSOR_OP_INDICES]

    @classmethod
    def get_output_names(cls) -> list[str]:
        """ONNX エクスポート時の出力テンソル名リストを取得する。

        Returns:
            list[str]: 出力テンソル名のリスト。
        """
        return [TENSOR_LOGITS]

    @classmethod
    def get_dynamic_axes(cls) -> dict[str, dict[int, str]]:
        """ONNX エクスポート時の動的軸定義辞書を取得する。

        Returns:
            dict[str, dict[int, str]]: 動的軸の辞書。
        """
        return {
            TENSOR_INPUT_IDS: {0: "batch_size", 1: "sequence_length"},
            TENSOR_ATTENTION_MASK: {0: "batch_size", 1: "sequence_length"},
            TENSOR_OP_INDICES: {0: "batch_size", 1: "num_options"},
            TENSOR_LOGITS: {0: "batch_size", 1: "num_options"},
        }
