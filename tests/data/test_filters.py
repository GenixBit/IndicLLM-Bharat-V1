from __future__ import annotations

import hashlib

import pytest

from bharat.data.exact_dedup import ExactDedupConfig, ExactDeduplicator
from bharat.data.fuzzy_dedup import FuzzyDedupConfig, FuzzyDeduplicator
from bharat.data.language_id import LanguageIDConfig, LanguageIdentifier
from bharat.data.normalization import NormalizationConfig, Normalizer
from bharat.data.pii import PIIConfig, PIIDetector
from bharat.data.processing import DataProcessor, ProcessingConfig, ProcessingDecision
from bharat.data.quality import QualityConfig, QualityDecision, QualityScorer
from bharat.data.safety_filter import SafetyFilter, SafetyFilterConfig

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalizer:
    def test_default_config(self):
        n = Normalizer()
        assert n.normalize("  Hello   World\r\n") == "Hello World"

    def test_nfc_normalization(self):
        composed = "\u00e9"
        decomposed = "e\u0301"
        n = Normalizer()
        assert n.normalize(decomposed) == composed

    def test_strip_disabled(self):
        config = NormalizationConfig(strip=False)
        n = Normalizer(config)
        result = n.normalize("  hello  ")
        assert "  " not in result or result == " hello "

    def test_lowercase(self):
        config = NormalizationConfig(lowercase=True)
        n = Normalizer(config)
        assert n.normalize("HELLO World") == "hello world"

    def test_remove_urls(self):
        config = NormalizationConfig(remove_urls=True)
        n = Normalizer(config)
        result = n.normalize("visit https://example.com now")
        assert "https://" not in result

    def test_remove_emails(self):
        config = NormalizationConfig(remove_emails=True)
        n = Normalizer(config)
        result = n.normalize("email me at test@example.com please")
        assert "test@" not in result or "test@example.com" not in result

    def test_remove_control_chars(self):
        n = Normalizer()
        result = n.normalize("hello\x00world\x1f")
        assert "\x00" not in result
        assert "\x1f" not in result

    def test_remove_zero_width(self):
        n = Normalizer()
        result = n.normalize("hello\u200bworld")
        assert "\u200b" not in result

    def test_collapse_whitespace(self):
        n = Normalizer()
        result = n.normalize("a   b\t\tc")
        assert result == "a b c"

    def test_max_length(self):
        config = NormalizationConfig(max_length=5)
        n = Normalizer(config)
        assert n.normalize("hello world") == "hello"

    def test_empty_text(self):
        n = Normalizer()
        assert n.normalize("") == ""

    def test_normalize_with_meta(self):
        n = Normalizer()
        result = n.normalize_with_meta("  Test  ")
        assert result.original == "  Test  "
        assert result.normalized == "Test"
        assert isinstance(result.config, NormalizationConfig)


# ---------------------------------------------------------------------------
# Language Identification
# ---------------------------------------------------------------------------


class TestLanguageIdentifier:
    def test_empty_text(self):
        li = LanguageIdentifier()
        result = li.identify("")
        assert result.language == "unknown"

    def test_too_short_latin_no_fallback(self):
        li = LanguageIdentifier()
        result = li.identify("hi")
        assert result.language == "unknown"
        assert result.method == "too_short"

    def test_short_indic_gets_script_fallback(self):
        li = LanguageIdentifier()
        result = li.identify("नमस्ते")
        assert result.method == "script_fallback"
        assert result.language == "hi"

    def test_short_indic_with_fallback_disabled(self):
        config = LanguageIDConfig(script_fallback=False)
        li = LanguageIdentifier(config)
        result = li.identify("नमस्ते")
        assert result.method == "too_short"

    def test_script_fallback_latin(self):
        li = LanguageIdentifier()
        result = li.identify("hello world this is a test of the language identifier")
        assert result.script == "Latin"
        assert result.language == "unknown"

    def test_script_fallback_devanagari(self):
        li = LanguageIdentifier()
        result = li.identify(
            "यह एक परीक्षण है जो देवनागरी लिपि में है और यह काफी लंबा है"
        )
        assert result.script == "DEVANAGARI"
        assert result.language == "hi"

    def test_bengali_text(self):
        li = LanguageIdentifier()
        result = li.identify(
            "এটি একটি পরীক্ষা যা বাংলা লিপিতে লেখা এবং এটি বেশ দীর্ঘ"
        )
        assert result.script == "BENGALI"
        assert result.language == "bn"

    def test_tamil_text(self):
        li = LanguageIdentifier()
        result = li.identify(
            "இது ஒரு சோதனை தமிழ் மொழியில் எழுதப்பட்ட உரை"
        )
        assert result.script == "TAMIL"
        assert result.language == "ta"

    def test_telugu_text(self):
        li = LanguageIdentifier()
        result = li.identify(
            "ఇది తెలుగు భాషలో రాసిన ఒక పరీక్ష వచనం"
        )
        assert result.script == "TELUGU"
        assert result.language == "te"

    def test_arabic_text(self):
        li = LanguageIdentifier()
        result = li.identify(
            "هذا اختبار للغة العربية وهو طويل بما فيه الكفاية"
        )
        assert result.script == "ARABIC"
        assert result.language == "ar"

    def test_mixed_script_weighted(self):
        li = LanguageIdentifier()
        result = li.identify(
            "यह देवनागरी है and some english words भी हैं"
        )
        assert result.script == "DEVANAGARI"

    def test_identify_batch(self):
        li = LanguageIdentifier()
        results = li.identify_batch(["short", "this is a longer english sentence for testing"])
        assert len(results) == 2

    def test_confidence_threshold_applied(self):
        config = LanguageIDConfig(confidence_threshold=0.99)
        li = LanguageIdentifier(config)
        text = "hello world this is a test of the language identifier function"
        result = li.identify(text)
        assert result.method in ("script_fallback", "unknown")


# ---------------------------------------------------------------------------
# Quality Scoring
# ---------------------------------------------------------------------------


class TestQualityScorer:
    def test_empty_text(self):
        qs = QualityScorer()
        score = qs.score("")
        assert score.overall == 0.0

    def test_high_quality_english(self):
        qs = QualityScorer()
        text = (
            "This is a reasonably long text that should pass quality checks. "
            "It has multiple sentences and paragraphs.\n\n"
            "The second paragraph continues the discussion with more content. "
            "Quality scoring should return a positive result for this input. "
            "We need enough words to pass the minimum thresholds.\n\n"
            "A third paragraph ensures we have enough lines and variety. "
            "This should be a high quality document overall."
        )
        decision = qs.evaluate(text)
        assert decision.is_quality
        assert decision.score > 0.3
        assert len(decision.reasons) == 0

    def test_short_text_not_quality(self):
        qs = QualityScorer()
        decision = qs.evaluate("short")
        assert not decision.is_quality
        assert "too_short" in decision.reasons or "too_few_words" in decision.reasons

    def test_excessive_punctuation(self):
        qs = QualityScorer()
        text = "hello!!! how are you??? this is great!!!! wow!!!!!"
        decision = qs.evaluate(text)
        assert isinstance(decision.is_quality, bool)

    def test_boilerplate_spam(self):
        qs = QualityScorer()
        text = "click here click here click here click here click here click here \n"
        decision = qs.evaluate(text)
        assert isinstance(decision.is_quality, bool)

    def test_repeated_text(self):
        qs = QualityScorer()
        text = "hello world hello world hello world hello world hello world " * 10
        decision = qs.evaluate(text)
        assert isinstance(decision.is_quality, bool)

    def test_single_line_paragraph(self):
        qs = QualityScorer()
        text = (
            "This is a single line paragraph. It is valid text but has only one line. "
            "It should still be evaluated correctly."
        )
        score = qs.score(text)
        assert isinstance(score.overall, float)

    def test_many_urls(self):
        qs = QualityScorer()
        text = "visit " + " https://example.com/" * 15
        decision = qs.evaluate(text)
        assert not decision.is_quality
        assert "too_many_urls" in decision.reasons or "too_few_words" in decision.reasons

    def test_custom_config_validation(self):
        with pytest.raises(ValueError, match="min_alpha_ratio"):
            QualityScorer(QualityConfig(min_alpha_ratio=-0.1))

    def test_min_chars_must_be_positive(self):
        with pytest.raises(ValueError, match="min_chars"):
            QualityScorer(QualityConfig(min_chars=0))

    def test_max_chars_less_than_min_raises(self):
        with pytest.raises(ValueError, match="max_chars"):
            QualityScorer(QualityConfig(min_chars=1000, max_chars=100))

    def test_min_words_must_be_positive(self):
        with pytest.raises(ValueError, match="min_words"):
            QualityScorer(QualityConfig(min_words=0))

    def test_features_present(self):
        qs = QualityScorer()
        score = qs.score("hello world foo bar baz")
        assert "chars" in score.features
        assert "words" in score.features
        assert "avg_word_len" in score.features

    def test_quality_decision_reason_codes(self):
        qs = QualityScorer()
        decision = qs.evaluate("a b")
        assert isinstance(decision, QualityDecision)
        assert isinstance(decision.reasons, tuple)
        assert len(decision.reasons) > 0

    def test_is_quality_delegates_to_evaluate(self):
        qs = QualityScorer()
        text = (
            "This is a reasonably long text that should pass quality checks. "
            "It has multiple sentences and paragraphs.\n\n"
            "The second paragraph continues the discussion with more content. "
            "Quality scoring should return a positive result for this input. "
        )
        assert qs.is_quality(text)


# ---------------------------------------------------------------------------
# Exact Deduplication
# ---------------------------------------------------------------------------


class TestExactDeduplicator:
    def test_add_new_document(self):
        dedup = ExactDeduplicator()
        assert dedup.add_document("hello world")
        assert dedup.seen_count == 1

    def test_add_duplicate_document(self):
        dedup = ExactDeduplicator()
        assert dedup.add_document("hello world")
        assert not dedup.add_document("hello world")
        assert dedup.seen_count == 1

    def test_is_duplicate(self):
        dedup = ExactDeduplicator()
        dedup.add_document("hello world")
        assert dedup.is_duplicate("hello world")
        assert not dedup.is_duplicate("different text")

    def test_filter(self):
        dedup = ExactDeduplicator()
        docs = ["a", "b", "a", "c", "b"]
        assert dedup.filter(docs) == ["a", "b", "c"]

    def test_reset(self):
        dedup = ExactDeduplicator()
        dedup.add_document("hello world")
        dedup.reset()
        assert dedup.seen_count == 0
        assert dedup.add_document("hello world")

    def test_empty_text(self):
        dedup = ExactDeduplicator()
        assert not dedup.add_document("")
        assert not dedup.is_duplicate("")

    def test_normalize_by_default(self):
        dedup = ExactDeduplicator()
        assert dedup.add_document("  Hello   World  ")
        assert dedup.is_duplicate("Hello World")

    def test_disable_normalization(self):
        config = ExactDedupConfig(normalize=False)
        dedup = ExactDeduplicator(config)
        assert dedup.add_document("  Hello   World  ")
        assert not dedup.is_duplicate("Hello World")
        assert dedup.is_duplicate("  Hello   World  ")

    def test_lowercase_normalization_in_config(self):
        norm_config = NormalizationConfig(lowercase=True)
        config = ExactDedupConfig(normalize=True, normalization_config=norm_config)
        dedup = ExactDeduplicator(config)
        assert dedup.add_document("HELLO WORLD")
        assert dedup.is_duplicate("hello world")

    def test_url_removal_in_normalization(self):
        norm_config = NormalizationConfig(remove_urls=True)
        config = ExactDedupConfig(normalize=True, normalization_config=norm_config)
        dedup = ExactDeduplicator(config)
        assert dedup.add_document("check https://example.com now")
        assert dedup.is_duplicate("check now")

    def test_email_removal_in_normalization(self):
        norm_config = NormalizationConfig(remove_emails=True, lowercase=True)
        config = ExactDedupConfig(normalize=True, normalization_config=norm_config)
        dedup = ExactDeduplicator(config)
        assert dedup.add_document("Email test@example.com for info")
        assert dedup.is_duplicate("email for info")

    def test_line_level_dedup(self):
        config = ExactDedupConfig(line_level=True)
        dedup = ExactDeduplicator(config)
        assert dedup.add_document("line1\nline2\nline1")
        assert dedup.seen_count == 2

    def test_line_level_seen_count(self):
        config = ExactDedupConfig(line_level=True)
        dedup = ExactDeduplicator(config)
        dedup.add_document("a\nb")
        dedup.add_document("b\nc")
        assert dedup.seen_count == 3

    def test_invalid_hash_algorithm(self):
        with pytest.raises(ValueError, match="hash algorithm"):
            ExactDeduplicator(ExactDedupConfig(hash_func="invalid_hash_xyz"))

    def test_valid_hash_md5(self):
        dedup = ExactDeduplicator(ExactDedupConfig(hash_func="md5"))
        assert dedup.add_document("hello world")
        assert dedup.is_duplicate("hello world")


# ---------------------------------------------------------------------------
# Fuzzy Deduplication
# ---------------------------------------------------------------------------


class TestFuzzyDeduplicator:
    def test_add_new_document(self):
        dedup = FuzzyDeduplicator()
        assert dedup.add_document("hello world this is a test document")
        assert dedup.seen_count == 1

    def test_add_identical_document(self):
        dedup = FuzzyDeduplicator()
        text = "hello world this is a test document for fuzzy dedup"
        assert dedup.add_document(text)
        assert not dedup.add_document(text)

    def test_hindi_near_duplicate(self):
        dedup = FuzzyDeduplicator(FuzzyDedupConfig(threshold=0.5))
        text_a = (
            "भारत एक महान देश है जहां विभिन्न संस्कृतियों का समावेश है "
            "और यहां की विविधता इसे विशेष बनाती है"
        )
        text_b = (
            "भारत एक महान देश है जहां विभिन्न संस्कृतियों का समावेश है "
            "और यहां की विविधता इसे खास बनाती है"
        )
        assert dedup.add_document(text_a)
        assert not dedup.add_document(text_b)

    def test_unrelated_hindi_not_duplicate(self):
        dedup = FuzzyDeduplicator()
        text_a = (
            "भारत एक महान देश है जहां विभिन्न संस्कृतियों का समावेश है"
        )
        text_b = (
            "तकनीकी विकास ने जीवन को सरल और सुविधाजनक बना दिया है"
        )
        assert dedup.add_document(text_a)
        assert dedup.add_document(text_b)
        assert dedup.seen_count == 2

    def test_tamil_text_dedup(self):
        dedup = FuzzyDeduplicator(FuzzyDedupConfig(threshold=0.5))
        text_a = (
            "தமிழ் மொழி மிகவும் பழமையான மொழிகளில் ஒன்றாகும் "
            "இது உலகம் முழுவதும் பல மில்லியன் மக்களால் பேசப்படுகிறது"
        )
        text_b = (
            "தமிழ் மொழி மிகவும் பழமையான மொழிகளில் ஒன்றாகும் "
            "இது உலகம் முழுவதும் பல மில்லியன் மக்களால் பேசப்படுகிறது"
        )
        assert dedup.add_document(text_a)
        assert not dedup.add_document(text_b)

    def test_mixed_english_hindi(self):
        dedup = FuzzyDeduplicator(FuzzyDedupConfig(threshold=0.5))
        text_a = "India is a great country भारत एक महान देश है"
        text_b = "India is a great country भारत एक महान राष्ट्र है"
        assert dedup.add_document(text_a)
        assert not dedup.add_document(text_b)

    def test_punctuation_only_input(self):
        dedup = FuzzyDeduplicator()
        assert not dedup.add_document("!!! ??? ...")
        assert dedup.seen_count == 0

    def test_empty_text(self):
        dedup = FuzzyDeduplicator()
        assert not dedup.add_document("")
        assert not dedup.is_duplicate("")

    def test_very_short_text(self):
        dedup = FuzzyDeduplicator()
        assert dedup.add_document("a b")
        assert dedup.seen_count == 1

    def test_invalid_n_gram_size(self):
        with pytest.raises(ValueError, match="n_gram_size"):
            FuzzyDeduplicator(FuzzyDedupConfig(n_gram_size=0))

    def test_invalid_num_permutations(self):
        with pytest.raises(ValueError, match="num_permutations"):
            FuzzyDeduplicator(FuzzyDedupConfig(num_permutations=0))

    def test_invalid_threshold_low(self):
        with pytest.raises(ValueError, match="threshold"):
            FuzzyDeduplicator(FuzzyDedupConfig(threshold=-0.1))

    def test_invalid_threshold_high(self):
        with pytest.raises(ValueError, match="threshold"):
            FuzzyDeduplicator(FuzzyDedupConfig(threshold=1.1))

    def test_is_duplicate(self):
        dedup = FuzzyDeduplicator()
        text = "hello world this is a test document for fuzzy dedup checking"
        dedup.add_document(text)
        assert dedup.is_duplicate(text)
        assert not dedup.is_duplicate("completely different topic altogether")

    def test_reset(self):
        dedup = FuzzyDeduplicator()
        text = "hello world this is a test document for fuzzy dedup"
        dedup.add_document(text)
        dedup.reset()
        assert dedup.seen_count == 0
        assert dedup.add_document(text)


# ---------------------------------------------------------------------------
# PII Detection
# ---------------------------------------------------------------------------


class TestPIIDetector:
    def test_detect_email(self):
        detector = PIIDetector()
        spans = detector.detect("contact me at user@example.com")
        assert len(spans) >= 1
        assert spans[0].pii_type == "email"

    def test_detect_phone(self):
        detector = PIIDetector()
        spans = detector.detect("call me at +1-555-123-4567")
        phone_spans = [s for s in spans if s.pii_type == "phone"]
        assert len(phone_spans) >= 1

    def test_detect_ip_valid(self):
        detector = PIIDetector()
        spans = detector.detect("server IP is 192.168.1.1")
        ip_spans = [s for s in spans if s.pii_type == "ip_address"]
        assert len(ip_spans) >= 1
        assert ip_spans[0].text == "192.168.1.1"

    def test_invalid_ip_rejected(self):
        detector = PIIDetector()
        spans = detector.detect("invalid IP 999.999.999.999")
        ip_spans = [s for s in spans if s.pii_type == "ip_address"]
        assert len(ip_spans) == 0

    def test_valid_credit_card_luhn(self):
        detector = PIIDetector()
        spans = detector.detect("card: 4111111111111111")
        cc_spans = [s for s in spans if s.pii_type == "credit_card"]
        assert len(cc_spans) >= 1

    def test_invalid_credit_card_luhn(self):
        detector = PIIDetector()
        spans = detector.detect("card: 1234567890123456")
        cc_spans = [s for s in spans if s.pii_type == "credit_card"]
        assert len(cc_spans) == 0

    def test_valid_cc_with_spaces_luhn(self):
        detector = PIIDetector()
        spans = detector.detect("card: 4111 1111 1111 1111")
        cc_spans = [s for s in spans if s.pii_type == "credit_card"]
        assert len(cc_spans) >= 1

    def test_detect_aadhaar(self):
        detector = PIIDetector()
        spans = detector.detect("aadhaar: 2345 6789 0123")
        aa_spans = [s for s in spans if s.pii_type == "aadhaar"]
        assert len(aa_spans) >= 1

    def test_detect_pan(self):
        detector = PIIDetector()
        spans = detector.detect("pan: ABCDE1234F")
        pan_spans = [s for s in spans if s.pii_type == "pan"]
        assert len(pan_spans) >= 1

    def test_detect_url_credentials(self):
        detector = PIIDetector()
        spans = detector.detect("url: https://user:pass@example.com")
        cred_spans = [s for s in spans if s.pii_type == "url_credentials"]
        assert len(cred_spans) >= 1

    def test_overlapping_phone_credit_card(self):
        detector = PIIDetector()
        spans = detector.detect("contact 4111111111111111")
        types = {s.pii_type for s in spans}
        assert "credit_card" in types

    def test_no_pii_in_clean_text(self):
        detector = PIIDetector()
        spans = detector.detect("hello world this is clean text")
        assert len(spans) == 0

    def test_has_pii(self):
        detector = PIIDetector()
        assert detector.has_pii("email: test@example.com")
        assert not detector.has_pii("clean text")

    def test_redact_overlapping_spans(self):
        detector = PIIDetector()
        result = detector.redact("contact test@example.com or 192.168.1.1")
        assert "test@example.com" not in result
        assert "192.168.1.1" not in result

    def test_redact_no_pii(self):
        detector = PIIDetector()
        result = detector.redact("clean text")
        assert result == "clean text"

    def test_enabled_types_filter(self):
        config = PIIConfig(enabled_types=("email",))
        detector = PIIDetector(config)
        spans = detector.detect("email: test@example.com and ip: 192.168.1.1")
        assert all(s.pii_type == "email" for s in spans)

    def test_min_confidence_filter(self):
        config = PIIConfig(min_confidence=0.9)
        detector = PIIDetector(config)
        spans = detector.detect("call 123")
        assert len(spans) == 0

    def test_empty_text(self):
        detector = PIIDetector()
        assert detector.detect("") == []
        assert not detector.has_pii("")
        assert detector.redact("") == ""

    def test_mixed_indian_pii(self):
        detector = PIIDetector()
        text = "Aadhaar 234567890123 and PAN ABCDE1234F"
        spans = detector.detect(text)
        types = {s.pii_type for s in spans}
        assert "aadhaar" in types or "pan" in types

    def test_custom_mask_char(self):
        config = PIIConfig(mask_char="#")
        detector = PIIDetector(config)
        result = detector.redact("email: test@example.com")
        assert "#" in result
        assert "test" not in result


# ---------------------------------------------------------------------------
# Safety Filter
# ---------------------------------------------------------------------------


class TestSafetyFilter:
    def test_clean_text_is_safe(self):
        sf = SafetyFilter()
        result = sf.classify("hello world this is a clean text")
        assert result.is_safe
        assert len(result.spans) == 0
        assert result.score == 1.0

    def test_hate_speech_detected(self):
        sf = SafetyFilter()
        result = sf.classify("you are a stupid moron")
        assert "hate_speech" in result.categories_violated
        assert result.score < 1.0

    def test_profanity_detected(self):
        sf = SafetyFilter()
        result = sf.classify("what the fuck")
        assert "profanity" in result.categories_violated
        assert isinstance(result.is_safe, bool)

    def test_violence_detected(self):
        sf = SafetyFilter()
        result = sf.classify("they plan to kill everyone")
        assert "violence" in result.categories_violated

    def test_sexual_content_detected(self):
        sf = SafetyFilter()
        result = sf.classify("this site has porn content")
        assert "sexual" in result.categories_violated

    def test_spam_detected(self):
        sf = SafetyFilter()
        result = sf.classify("click here to subscribe now")
        assert "spam" in result.categories_violated

    def test_safe_educational_content(self):
        sf = SafetyFilter()
        text = "The biology textbook discussed sexual reproduction in plants and animals."
        result = sf.classify(text)
        assert result.is_safe

    def test_disabled_categories(self):
        config = SafetyFilterConfig(enabled_categories=("violence",))
        sf = SafetyFilter(config)
        result = sf.classify("you are an idiot")
        assert result.is_safe

    def test_empty_text_is_safe(self):
        sf = SafetyFilter()
        result = sf.classify("")
        assert result.is_safe
        assert result.score == 1.0

    def test_reason_codes_present(self):
        sf = SafetyFilter()
        result = sf.classify("kill the idiot")
        assert len(result.reasons) > 0

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            SafetyFilter(SafetyFilterConfig(threshold=1.5))

    def test_invalid_min_confidence_raises(self):
        with pytest.raises(ValueError, match="min_confidence"):
            SafetyFilter(SafetyFilterConfig(min_confidence=-0.1))

    def test_filter_method(self):
        sf = SafetyFilter()
        texts = ["hello world", "you are an idiot", "clean text"]
        result = sf.filter(texts)
        assert "hello world" in result

    def test_reasons_format(self):
        sf = SafetyFilter()
        result = sf.classify("kill the idiot")
        for reason in result.reasons:
            assert ":" in reason


# ---------------------------------------------------------------------------
# Data Processing Pipeline
# ---------------------------------------------------------------------------


class TestDataProcessor:
    def test_process_clean_text(self):
        processor = DataProcessor()
        decision = processor.process("This is a clean text that should be accepted.")
        assert isinstance(decision, ProcessingDecision)
        assert isinstance(decision.accepted, bool)
        assert isinstance(decision.reasons, tuple)

    def test_process_empty_text(self):
        processor = DataProcessor()
        decision = processor.process("")
        assert not decision.accepted

    def test_process_batch(self):
        processor = DataProcessor()
        texts = ["first document.", "second document here.", ""]
        decisions = processor.process_batch(texts)
        assert len(decisions) == 3
        assert all(isinstance(d, ProcessingDecision) for d in decisions)

    def test_deterministic(self):
        config = ProcessingConfig()
        p1 = DataProcessor(config)
        p2 = DataProcessor(config)
        d1 = p1.process("hello world test document")
        d2 = p2.process("hello world test document")
        assert d1.accepted == d2.accepted
        assert d1.reasons == d2.reasons

    def test_reset_dedup(self):
        processor = DataProcessor()
        processor.process("hello world")
        processor.reset_dedup()
        decision = processor.process("hello world")
        assert isinstance(decision, ProcessingDecision)

    def test_pipeline_contains_all_fields(self):
        processor = DataProcessor()
        decision = processor.process("test document for pipeline processing")
        assert hasattr(decision, "accepted")
        assert hasattr(decision, "normalized_text")
        assert hasattr(decision, "language")
        assert hasattr(decision, "quality_score")
        assert hasattr(decision, "reasons")
        assert hasattr(decision, "pii_spans")
        assert hasattr(decision, "safety_spans")

    def test_custom_config(self):
        config = ProcessingConfig()
        processor = DataProcessor(config)
        decision = processor.process("test")
        assert isinstance(decision, ProcessingDecision)
