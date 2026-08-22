from __future__ import annotations

from bharat.verification.verifier import FactVerificationEngine, VerificationAssessment


class TestFactVerifier:
    def test_verify_claim_with_consensus(self):
        verifier = FactVerificationEngine()
        sources = [
            {
                "title": "ISRO Press Release",
                "url": "https://isro.gov.in/pr1",
                "publisher": "ISRO",
                "published_date": "2026-03-01",
                "extracted_text": "Chandrayaan-3 achieved lunar landing at South Pole with 100% mission objectives met.",
            },
            {
                "title": "PIB India",
                "url": "https://pib.gov.in/pr2",
                "publisher": "PIB",
                "published_date": "2026-03-02",
                "extracted_text": "Chandrayaan-3 successfully completed all experiments on the lunar surface.",
            },
        ]
        assessment = verifier.verify_claim(
            "Chandrayaan-3 successfully landed on the lunar south pole.", sources
        )
        assert isinstance(assessment, VerificationAssessment)
        assert assessment.is_verified
        assert assessment.confidence_score >= 0.75
        assert len(assessment.citations) == 2

        grounded = verifier.format_grounded_response(
            "The mission was successful.", assessment.citations
        )
        assert "Verified Citations" in grounded
