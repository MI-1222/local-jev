"""Banking77 データセットコンバータモジュール。

77 種類の銀行・金融ドメイン意図分類データを、Jev 統一スキーマ (Choice 型)
へ正規化・変換する。ラベルのスネークケース名を自然言語説明文へ変換し、
過大候補数対策として動的サブサンプリングをサポートする。
"""

import random
from collections.abc import Iterator
from typing import Any

from datasets import ClassLabel, Dataset, load_dataset

from data.converters.base import BaseDatasetConverter
from data.prompt_pool import sample_instruction
from data.schema import QuestionType, UnifiedSample

# 代表的ラベルの自然言語説明辞書 (未指定ラベルはアンダースコア置換でフォールバック)
BANKING77_LABEL_DESCRIPTIONS: dict[str, str] = {
    "activate_my_card": "Activating a newly received debit or credit card",
    "age_limit": "Inquiring about age requirements and eligibility limits",
    "apple_pay_or_google_pay": "Setting up or troubleshooting Apple Pay or Google Pay",
    "atm_support": "Assistance with ATM withdrawal, deposit, or machine errors",
    "automatic_top_up": "Configuring automatic account top-up or recurring funding",
    "balance_not_updated_after_cheque_or_cash_deposit": "Reporting delayed balance update after cheque or cash deposit",
    "balance_not_updated_after_bank_transfer": "Reporting delayed balance update after bank transfer",
    "beneficiary_not_allowed": "Troubleshooting issues where adding a transfer beneficiary is blocked",
    "cancel_transfer": "Requesting cancellation or reversal of an outbound transfer",
    "card_about_to_expire": "Instructions or renewal status for a card nearing expiration",
    "card_acceptance": "Inquiring about card acceptance at specific merchants or regions",
    "card_arrival": "Checking delivery status and arrival timeline of a new card",
    "card_delivery_estimate": "Estimated timeframe and shipping details for physical card delivery",
    "card_linking": "Help with linking or adding an existing card to the account",
    "card_not_working": "Troubleshooting physical card chip, stripe, or terminal failure",
    "card_payment_fee_charged": "Inquiring about unexpected merchant or transaction card fees",
    "card_payment_not_recognised": "Reporting an unrecognized or unauthorized card transaction",
    "card_payment_wrong_exchange_rate": "Disputing foreign exchange rates applied to a card payment",
    "card_swallowed": "Reporting an ATM that retained or swallowed the card",
    "cash_withdrawal_charge": "Questions about ATM cash withdrawal fees and limits",
    "cash_withdrawal_not_recognised": "Reporting an unrecognized cash withdrawal transaction",
    "change_pin": "Procedure for updating or changing the card PIN",
    "compromised_card": "Reporting a compromised, skimmed, or insecure card",
    "contactless_not_working": "Troubleshooting contactless NFC tap-to-pay failure",
    "country_support": "Checking list of supported countries and cross-border availability",
    "declined_card_payment": "Investigating why a debit or credit card transaction was declined",
    "declined_cash_withdrawal": "Investigating why an ATM cash withdrawal was declined",
    "declined_transfer": "Investigating why a bank transfer was rejected or declined",
    "direct_debit_payment_not_recognised": "Reporting an unrecognized direct debit withdrawal",
    "disposable_virtual_card": "Creating or using single-use disposable virtual cards",
    "edit_personal_details": "Updating user address, phone number, or profile details",
    "exchange_charge": "Fees associated with currency exchange transactions",
    "exchange_rate": "Current live foreign exchange conversion rates",
    "exchange_via_app": "Executing in-app currency exchanges between balance pots",
    "extra_charge_on_statement": "Inquiring about unexpected surcharges on bank statements",
    "failed_transfer": "Investigating a failed or bounced bank transfer",
    "fiat_currency_support": "Inquiring about supported fiat currencies and account balances",
    "get_disposable_virtual_card": "Generating a single-use virtual card for secure online purchases",
    "get_physical_card": "Ordering a new physical payment card",
    "getting_spare_card": "Ordering an additional spare card for travel or backup",
    "getting_virtual_card": "Creating a permanent multi-use virtual card",
    "lost_or_stolen_card": "Reporting a lost, stolen, or misplaced physical card",
    "lost_or_stolen_phone": "Reporting a lost or stolen mobile phone with banking app access",
    "order_physical_card": "Step-by-step guidance on ordering a physical card",
    "passcode_forgotten": "Recovering or resetting forgotten app passcode or credentials",
    "pending_card_payment": "Status and processing timeframe of pending card authorizations",
    "pending_cash_withdrawal": "Questions regarding pending ATM cash withdrawals",
    "pending_top_up": "Status and clearing timeframe of pending top-up funds",
    "pending_transfer": "Status and delivery timeline of pending outbound bank transfers",
    "pin_blocked": "Unblocking card PIN after entering incorrect numbers repeatedly",
    "receiving_money": "Instructions on receiving domestic and international transfers",
    "request_refund": "Requesting a refund or chargeback for an eligible transaction",
    "reverted_card_payment_": "Clarification on reverted or refunded card authorizations",
    "supported_cards_and_currencies": "Supported card networks (Visa, Mastercard) and currency pairs",
    "terminate_account": "Procedure and requirements for closing or terminating the bank account",
    "top_up_by_bank_transfer_charge": "Fees incurred when topping up via bank wire or transfer",
    "top_up_by_card_charge": "Fees incurred when topping up using another debit/credit card",
    "top_up_by_cash_or_cheque": "Inquiring if cash or cheque deposits are accepted for top-up",
    "top_up_failed": "Troubleshooting failed account top-up or declined deposit",
    "top_up_limits": "Daily, weekly, or annual account funding limits",
    "top_up_reverted": "Understanding why a recent account top-up was reversed",
    "topping_up_by_card": "Guidance on adding funds instantly using an external card",
    "transaction_charged_twice": "Reporting duplicate charges for a single purchase",
    "transfer_fee_charged": "Questions regarding domestic or international transfer fees",
    "transfer_into_account": "Account details and routing numbers for inbound transfers",
    "transfer_not_received_by_recipient": "Tracking delayed transfers not yet received by the recipient",
    "transfer_timing": "Estimated delivery schedules and cutoff times for bank transfers",
    "unable_to_verify_identity": "Assistance when identity verification (KYC) fails",
    "verify_my_identity": "Steps required to complete passport or ID verification",
    "verify_source_of_funds": "Providing documentation to verify the source of account funds",
    "verify_top_up": "Submitting verification details for security-flagged top-up deposits",
    "virtual_card_not_working": "Troubleshooting declined virtual card transactions",
    "visa_or_mastercard": "Network specifications regarding Visa and Mastercard issuance",
    "why_verify_identity": "Explanations on regulatory anti-money laundering (AML) KYC requirements",
    "wrong_amount_of_cash_received": "Reporting discrepancies in cash dispensed from an ATM",
    "wrong_exchange_rate_for_cash_withdrawal": "Disputing foreign exchange rate applied during foreign ATM withdrawal",
}


def clean_banking77_label(raw_label: str) -> str:
    """生ラベル文字列を自然言語の説明文に変換する。

    Args:
        raw_label (str): 生のスネークケースラベル名。

    Returns:
        str: 自然言語化された説明文。
    """
    clean_name = raw_label.rstrip("_")
    if clean_name in BANKING77_LABEL_DESCRIPTIONS:
        return BANKING77_LABEL_DESCRIPTIONS[clean_name]
    return clean_name.replace("_", " ").capitalize()


class Banking77Converter(BaseDatasetConverter):
    """Banking77 データセットコンバータ。

    Attributes:
        max_negative_options (int | None): 訓練時に動的抽出する負例候補の最大数。
            None の場合は全候補を展開する。
    """

    def __init__(
        self,
        max_negative_options: int | None = 7,
        seed: int = 42,
    ) -> None:
        """Banking77 コンバータを初期化する。

        Args:
            max_negative_options (int | None): 訓練時サブサンプリングする負例候補数。
            seed (int): 乱数シード。
        """
        super().__init__(seed=seed)
        self.max_negative_options = max_negative_options

    def convert_dataset(
        self,
        dataset: Dataset,
        split: str,
    ) -> Iterator[UnifiedSample]:
        """与えられた Hugging Face Dataset を走査して UnifiedSample を生成する。

        Args:
            dataset (Dataset): 対象の Dataset オブジェクト。
            split (str): スプリット名。

        Yields:
            Iterator[UnifiedSample]: 変換後の統一サンプル列。
        """
        rng = random.Random(self.seed)
        features: Any = dataset.features.get("label")
        if isinstance(features, ClassLabel) and features.names:
            raw_label_names: list[str] = list(features.names)
        else:
            raw_label_names = list(BANKING77_LABEL_DESCRIPTIONS.keys())

        all_descriptions = {lbl: clean_banking77_label(lbl) for lbl in raw_label_names}

        for idx, row in enumerate(dataset):
            if row.get("label_text"):
                target_id = str(row["label_text"])
            elif isinstance(row["label"], int) and 0 <= row["label"] < len(
                raw_label_names
            ):
                target_id = raw_label_names[row["label"]]
            else:
                target_id = str(row["label"])
            text = row["text"]

            # 訓練時かつ max_negative_options が指定されている場合は負例をサンプリング
            if (
                split == "train"
                and self.max_negative_options is not None
                and raw_label_names
                and self.max_negative_options < len(raw_label_names) - 1
            ):
                negatives = [lbl for lbl in raw_label_names if lbl != target_id]
                selected_negs = rng.sample(negatives, self.max_negative_options)
                candidate_ids = [target_id] + selected_negs
            else:
                candidate_ids = (
                    list(raw_label_names)
                    if raw_label_names
                    else list(all_descriptions.keys())
                )

            criteria = {
                cid: all_descriptions.get(cid, clean_banking77_label(cid))
                for cid in candidate_ids
            }
            instructions = sample_instruction("intent", rng=rng)

            yield UnifiedSample(
                dataset_name="banking77",
                sample_id=f"banking77_{split}_{idx}",
                question_type=QuestionType.CHOICE,
                state=text,
                instructions=instructions,
                criteria=criteria,
                target=target_id,
                metadata={
                    "split": split,
                    "original_label_id": row["label"],
                    "num_options": len(criteria),
                },
            )

    def convert_split(self, split: str) -> Iterator[UnifiedSample]:
        """Hugging Face から Banking77 をロードして変換する。

        Parquet 形式の mteb/banking77 を優先してロードし、
        レガシースクリプト非推奨環境 (datasets>=5.0) に対応する。
        Banking77 は検証スプリットを持たないため、validation 指定時は test を使用する。

        Args:
            split (str): スプリット名。

        Yields:
            Iterator[UnifiedSample]: 統一サンプル列。
        """
        hf_split = "test" if split in ["validation", "val"] else split
        try:
            ds = load_dataset("mteb/banking77", split=hf_split)
        except (RuntimeError, ValueError, OSError):
            ds = load_dataset("PolyAI/banking77", split=hf_split)
        assert isinstance(ds, Dataset)
        yield from self.convert_dataset(ds, split=split)
