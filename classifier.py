"""
classifier.py - AI-Assisted Ticket Classification Pipeline for Vireo Audio.

Predicts support category and assigned team from customer message and agent closing notes.
Preserves original category and provides confidence scores and explainable rationale.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# 11 Standard Vireo Support Categories
VALID_CATEGORIES = [
    "Delivery & Shipping",
    "Billing & Payments",
    "Returns & Refunds",
    "Warranty & Repair",
    "Charging & Battery",
    "Connectivity",
    "Audio Quality",
    "App & Firmware",
    "Account & Login",
    "Product Enquiry",
    "Other"
]

# Domain Pattern Rules (Category, Regex Pattern, Rationale)
DOMAIN_RULES: List[Tuple[str, str, str]] = [
    (
        "Delivery & Shipping",
        r"(?:shipment not (?:received|rcvd)|order not delivered|ord not delivered|dlvry delayed|delivery delayed|rto confirmed|re-shipped|transit damage|undelivered|package not (?:received|delivered)|parcel stuck|tracking not updating|awb with (?:courier|crr)|address update|wrong address|courier conf|courier partner|where is my order|dispatch status|already dispatched|out for delivery|wrong item delivered|wrong variant received|incorrect product shipped)",
        "Delivery tracking, transit delay, or shipment fulfillment issue"
    ),
    (
        "Billing & Payments",
        r"(?:duplicate payment|amount deducted without (?:ord|order)|payment debited,? no order|failed order after payment|card charged (?:two|twice)|money debited but order not|gst invoice|invoice (?:query|not downloading|emailed|pdf)|regenerated invoice|tax bill|payment gateway|promo code invalid|discount not applied|coupon validity|upi debited|double charge|charged extra|refund of duplicate)",
        "Payment gateway charge, invoice / GST bill, or discount coupon query"
    ),
    (
        "Returns & Refunds",
        r"(?:reverse pickup|pickup pending|pkp (?:not done|missed|done)|pickup awb|refund not credited|refund delay|refund pending|refund status on pg|refund initiated to source|refund approved|arn shared|return qc|cancel return|return request|return received|want to return|send it back)",
        "Return pickup request, return QC, or refund processing status"
    ),
    (
        "Warranty & Repair",
        r"(?:rma status|rma number|warranty claim|service cent(?:re|er)|repair status|escalated to (?:warranty|wty)|out of warranty|send unit for repair|warranty replacement|hardware fault)",
        "Hardware warranty claim, RMA processing, or repair request"
    ),
    (
        "Charging & Battery",
        r"(?:battery draining fast|poor battery backup|rapid battery drain|left (?:earbud|bud) not (?:taking )?charge|right (?:earbud|bud) not (?:taking )?charge|case not charging|charging case|pin cleaning|green light in case|battery life|won\'t charge|battery backup|no charge in case|battery drop|charging issue|power on)",
        "Battery life degradation, case charging, or power issue"
    ),
    (
        "Connectivity",
        r"(?:pairing failure|device not discoverable|intermittent disconnects|bt dropouts|bluetooth not connecting|bluetooth pairing|cannot pair|re-pair successful|disconnecting constantly|bluetooth scan|multipoint|drop connection|unpair)",
        "Bluetooth pairing, wireless connection drops, or discoverability issue"
    ),
    (
        "Audio Quality",
        r"(?:mic issue|mic not working|static noise|crackling|distortion|frying sound|low volume|no audio one side|audio drop|sound quality|one side louder|background noise in mic|muffled sound|bass issue|anc noise)",
        "Audio output distortion, microphone quality, or volume imbalance"
    ),
    (
        "App & Firmware",
        r"(?:app crash|app crashing|update hang|firmware update|fw update|fw rollout|bricked|vireo app|beta build|app cannot connect|device settings crash|eq settings)",
        "Companion app crash, firmware update failure, or software glitch"
    ),
    (
        "Account & Login",
        r"(?:login issue|unable to log in|otp not (?:received|coming)|account unlocked|password reset|locked out of (?:my )?account|login code|change phone number|registered number)",
        "Account authentication, OTP delivery, or profile access issue"
    ),
    (
        "Product Enquiry",
        r"(?:pre-sales query|compatibility query|spec sheet|shared compatibility|query answered|specifications|waterproof|water resistance|ipx rating|battery spec|will it work with)",
        "Product specifications, compatibility inquiry, or pre-purchase question"
    ),
]


def map_category_to_team(category: str, channel: str = "chat") -> str:
    """
    Maps support category and channel to responsible team as defined in
    support-policy.pdf Section 6.
    """
    category = str(category).strip()
    channel = str(channel).strip().lower()

    if category == "Delivery & Shipping":
        return "Logistics"
    elif category == "Billing & Payments":
        return "Billing"
    elif category == "Returns & Refunds":
        return "Returns Desk"
    elif category == "Warranty & Repair":
        return "Escalations & Warranty"
    else:
        # Tier 1 Frontline categories (Product Enquiry, Account & Login, Connectivity,
        # Charging & Battery, Audio Quality, App & Firmware, Other)
        if channel in ["chat", "social"]:
            return "Chat Frontline"
        elif channel == "email":
            return "Email Frontline"
        elif channel == "voice":
            return "Voice Frontline"
        return "Chat Frontline"


class TicketClassifier:
    """
    Hybrid Rule + Probabilistic ML Classifier for support tickets.
    """

    def __init__(self):
        self.rules = DOMAIN_RULES
        self.ml_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=10000,
                sublinear_tf=True
            )),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced"
            ))
        ])
        self.is_fitted = False

    def _extract_text(self, customer_message: Any, agent_notes: Any) -> str:
        msg = str(customer_message) if pd.notnull(customer_message) else ""
        notes = str(agent_notes) if pd.notnull(agent_notes) else ""
        return f"{msg} {notes}".strip()

    def _match_rule(self, text: str) -> Optional[Tuple[str, float, str]]:
        text_lower = text.lower()
        for cat, pat, reason in self.rules:
            m = re.search(pat, text_lower)
            if m:
                matched_kw = m.group(0)
                return cat, 0.95, f"{reason} (keyword: '{matched_kw}')"
        return None

    def fit_from_dataframe(self, df: pd.DataFrame) -> "TicketClassifier":
        """
        Trains the statistical classifier using high-confidence rule matches as seeds.
        """
        texts = [self._extract_text(r.get("customer_message"), r.get("agent_notes")) for _, r in df.iterrows()]
        
        seed_labels = []
        seed_indices = []
        for idx, t in enumerate(texts):
            match = self._match_rule(t)
            if match:
                seed_labels.append(match[0])
                seed_indices.append(idx)

        if len(seed_indices) > 50:
            seed_texts = [texts[i] for i in seed_indices]
            self.ml_pipeline.fit(seed_texts, seed_labels)
            self.is_fitted = True

        return self

    def classify_single(
        self,
        customer_message: str,
        agent_notes: str = "",
        channel: str = "chat",
        ticket_id: str = "TK-NEW",
        original_category: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Classifies a single ticket and returns all required output fields.
        """
        text = self._extract_text(customer_message, agent_notes)
        
        # 1. Check high-confidence rule
        rule_match = self._match_rule(text)
        if rule_match:
            pred_cat, conf, reason = rule_match
        elif self.is_fitted and text.strip():
            # 2. Use ML model
            probs = self.ml_pipeline.predict_proba([text])[0]
            classes = self.ml_pipeline.classes_
            best_idx = np.argmax(probs)
            pred_cat = classes[best_idx]
            conf = float(probs[best_idx])
            reason = f"AI/ML semantic text classification ({pred_cat}) based on message and notes context"
        else:
            # Fallback
            pred_cat = "Other"
            conf = 0.50
            reason = "Default fallback due to limited text context"

        pred_team = map_category_to_team(pred_cat, channel)

        return {
            "ticket_id": ticket_id,
            "original_category": original_category,
            "predicted_category": pred_cat,
            "predicted_team": pred_team,
            "confidence": round(conf, 3),
            "short_reason": reason
        }

    def batch_classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch classifies a DataFrame of tickets, appending predicted fields.
        """
        if not self.is_fitted:
            self.fit_from_dataframe(df)

        results = []
        for _, row in df.iterrows():
            res = self.classify_single(
                customer_message=row.get("customer_message", ""),
                agent_notes=row.get("agent_notes", ""),
                channel=row.get("channel", "chat"),
                ticket_id=row.get("ticket_id", ""),
                original_category=row.get("category", "Unknown")
            )
            results.append(res)

        res_df = pd.DataFrame(results)
        
        output_df = df.copy()
        output_df["predicted_category"] = res_df["predicted_category"].values
        output_df["predicted_team"] = res_df["predicted_team"].values
        output_df["confidence"] = res_df["confidence"].values
        output_df["short_reason"] = res_df["short_reason"].values

        return output_df
