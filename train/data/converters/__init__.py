"""Jev データセットコンバータパッケージ。

Banking77, CLINC150, MNLI, AG News などの公開 NLP コーパスを
統一スキーマ `UnifiedSample` へ変換するコンバータ群を提供する。
"""

from data.converters.ag_news import AGNewsConverter
from data.converters.banking77 import Banking77Converter
from data.converters.base import BaseDatasetConverter
from data.converters.clinc150 import Clinc150Converter
from data.converters.jglue_jcommonsenseqa import JGlueJCommonsenseQAConverter
from data.converters.jglue_jnli import JGlueJNLIConverter
from data.converters.jglue_jsts import JGlueJSTSConverter
from data.converters.jglue_marc_ja import JGlueMarcJaConverter
from data.converters.mnli import MNLIConverter
from data.converters.sst5 import SST5Converter

__all__ = [
    "AGNewsConverter",
    "Banking77Converter",
    "BaseDatasetConverter",
    "Clinc150Converter",
    "JGlueJCommonsenseQAConverter",
    "JGlueJNLIConverter",
    "JGlueJSTSConverter",
    "JGlueMarcJaConverter",
    "MNLIConverter",
    "SST5Converter",
]
