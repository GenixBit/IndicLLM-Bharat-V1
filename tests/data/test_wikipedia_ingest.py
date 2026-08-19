from __future__ import annotations

from pathlib import Path

from bharat.data.wikipedia_ingest import (
    clean_wikipedia_text,
    extract_wikipedia_articles,
    ingest_and_pack_wikipedia,
    is_valid_indic_script,
)
from scripts.ingest_wikipedia import main as ingest_wiki_main
from scripts.ingest_wikipedia import parse_args


class TestWikipediaIngest:
    def test_clean_wikipedia_text(self):
        dirty = "<p>India is a sovereign country.[[Category:India|*]] {{cite web|url=example.com}} [1] Contact test@test.com</p>"
        cleaned = clean_wikipedia_text(dirty)
        assert "<p>" not in cleaned
        assert "[1]" not in cleaned
        assert "[EMAIL]" in cleaned
        assert "sovereign country" in cleaned

    def test_is_valid_indic_script(self):
        hindi_text = "भारत गणराज्य दक्षिण एशिया में स्थित एक सम्प्रभु देश है।"
        assert is_valid_indic_script(hindi_text, "hi", min_chars=20)

        english_only = "This is entirely english text without devanagari characters."
        assert not is_valid_indic_script(english_only, "hi", min_chars=20)

    def test_extract_wikipedia_articles(self):
        articles = extract_wikipedia_articles(languages=["hi", "bn", "ta", "en"])
        assert len(articles) > 0
        langs = {a["lang"] for a in articles}
        assert "hi" in langs
        assert "en" in langs

    def test_ingest_and_pack_wikipedia(self, tmp_path: Path):
        res = ingest_and_pack_wikipedia(
            output_dir=tmp_path / "wiki_shards",
            languages=["hi", "bn", "ta", "en"],
            max_docs_per_lang=2,
            max_tokens_per_shard=1000,
        )

        assert res.total_articles > 0
        assert len(res.shards_written) > 0
        assert res.shards_written[0].is_file()

    def test_cli_parse_args(self):
        args = parse_args(["--langs", "hi,bn", "--max-docs-per-lang", "10"])
        assert args.langs == "hi,bn"
        assert args.max_docs_per_lang == 10

    def test_cli_main(self, tmp_path: Path):
        code = ingest_wiki_main(
            [
                "--langs",
                "hi,en",
                "--output-dir",
                str(tmp_path / "cli_wiki"),
                "--max-docs-per-lang",
                "2",
            ]
        )
        assert code == 0
        assert len(list((tmp_path / "cli_wiki").glob("*.bin"))) > 0
