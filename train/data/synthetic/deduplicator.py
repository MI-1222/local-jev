"""意味的重複排除 (Semantic Deduplication) モジュール。

State テキストの埋め込みベクトルを算出し、既存データセットとのコサイン類似度が
閾値 (デフォルト: 0.92) 以上の重複事例を高速に検出・棄却する。
"""

import math
from collections import Counter


class TextEmbeddingDeduplicator:
    """文字 n-gram 頻度ベースの高次元 TF-IDF 埋め込みを用いた重複排除エンジン。

    外部埋め込みモデルのダウンロード待ちなしに決定論的かつ高速に動作し、
    文脈の構文的・意味的重複度をコサイン類似度として測定する。

    Attributes:
        threshold (float): 重複と判定するコサイン類似度閾値 (デフォルト: 0.92)。
        n_gram (int): 分割する n-gram 長 (デフォルト: 3)。
    """

    def __init__(self, threshold: float = 0.92, n_gram: int = 3) -> None:
        """重複排除器を初期化する。

        Args:
            threshold (float): コサイン類似度閾値。
            n_gram (int): n-gram サイズ。
        """
        self.threshold = threshold
        self.n_gram = n_gram
        self._index: list[tuple[str, Counter[str], float]] = []

    def _extract_vector(self, text: str) -> tuple[Counter[str], float]:
        """テキストから n-gram カウンタと L2 ノルムを計算する。

        Args:
            text (str): 入力テキスト。

        Returns:
            tuple[Counter[str], float]: n-gram 頻度カウンタと L2 ノルム。
        """
        cleaned = "".join(text.split())
        ngrams: list[str] = [
            cleaned[i : i + self.n_gram]
            for i in range(max(1, len(cleaned) - self.n_gram + 1))
        ]
        counter = Counter(ngrams)
        norm_sq = sum(v * v for v in counter.values())
        norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
        return counter, norm

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """2つのテキスト間のコサイン類似度を計算する。

        Args:
            text_a (str): 比較元テキスト。
            text_b (str): 比較先テキスト。

        Returns:
            float: 0.0〜1.0 のコサイン類似度。
        """
        counter_a, norm_a = self._extract_vector(text_a)
        counter_b, norm_b = self._extract_vector(text_b)

        dot_product = 0.0
        if len(counter_a) < len(counter_b):
            for k, val_a in counter_a.items():
                if k in counter_b:
                    dot_product += val_a * counter_b[k]
        else:
            for k, val_b in counter_b.items():
                if k in counter_a:
                    dot_product += val_b * counter_a[k]

        return dot_product / (norm_a * norm_b)

    def is_duplicate(
        self,
        sample_id: str,
        text: str,
        add_if_unique: bool = True,
    ) -> tuple[bool, float, str | None]:
        """テキストが既存インデックスと重複しているかを判定する。

        Args:
            sample_id (str): サンプル識別子。
            text (str): 判定対象テキスト。
            add_if_unique (bool): 重複していない場合にインデックスへ追加するか。

        Returns:
            tuple[bool, float, str | None]:
                - 重複フラグ (True なら重複)。
                - 最大コサイン類似度。
                - 最も類似していた既存サンプルの ID (存在しない場合は None)。
        """
        counter, norm = self._extract_vector(text)
        max_sim = 0.0
        most_similar_id: str | None = None

        for existing_id, existing_counter, existing_norm in self._index:
            dot = 0.0
            for k, val in counter.items():
                if k in existing_counter:
                    dot += val * existing_counter[k]
            sim = dot / (norm * existing_norm)
            if sim > max_sim:
                max_sim = sim
                most_similar_id = existing_id
                if max_sim >= self.threshold:
                    return True, max_sim, most_similar_id

        if add_if_unique:
            self._index.append((sample_id, counter, norm))

        return False, max_sim, most_similar_id

    def clear(self) -> None:
        """インデックスを初期化する。"""
        self._index.clear()
