"""Tests for the SparseRetriever (BM25) adapter."""

from __future__ import annotations

import numpy as np
import pytest

from cuecard.models import Index, Provenance, Rule
from cuecard.retrieval.fusion import ScoredCandidate
from cuecard.retrieval.sparse import BM25Okapi, SparseRetriever, _tokenize


def _make_rules(count: int) -> tuple[Rule, ...]:
    """Create count distinct rules."""
    return tuple(
        Rule(
            text=f"Rule {i}",
            provenance=Provenance(file="/tmp/r.txt", line_start=i, line_end=i),
        )
        for i in range(count)
    )


def _make_index(
    rules: tuple[Rule, ...],
    bm25_corpus: tuple[str, ...] | None,
    rule_map: tuple[int, ...] | None = None,
) -> Index:
    """Build a minimal Index with bm25_corpus for sparse testing."""
    n = len(rule_map) if rule_map else len(rules)
    emb = np.zeros((n, 8), dtype=np.float32)
    return Index(
        embeddings=emb,
        rules=rules,
        model_name="test",
        dim=8,
        sources={},
        rule_map=rule_map,
        bm25_corpus=bm25_corpus,
    )


class TestTokenize:
    """Tokenizer splits on whitespace + code punctuation."""

    def test_simple_words(self) -> None:
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_dot_splitting(self) -> None:
        assert _tokenize("aiohttp.ClientSession") == ["aiohttp", "clientsession"]

    def test_paren_splitting(self) -> None:
        assert _tokenize("eval()") == ["eval"]

    def test_bracket_splitting(self) -> None:
        assert _tokenize("list[str]") == ["list", "str"]

    def test_curly_brace_splitting(self) -> None:
        assert _tokenize("dict{}") == ["dict"]

    def test_mixed(self) -> None:
        tokens = _tokenize("os.path.join(a, b)")
        assert tokens == ["os", "path", "join", "a,", "b"]

    def test_empty_string(self) -> None:
        assert _tokenize("") == []

    def test_only_delimiters(self) -> None:
        assert _tokenize("... ()") == []


class TestBM25Okapi:
    """Inline BM25Okapi implementation tests."""

    def test_basic_scoring(self) -> None:
        corpus = ("the cat sat on the mat", "the dog played in the yard")
        bm25 = BM25Okapi(corpus)
        scores = bm25.score("cat")
        assert scores[0] > scores[1]  # "cat" appears in doc 0

    def test_no_match(self) -> None:
        corpus = ("hello world", "foo bar")
        bm25 = BM25Okapi(corpus)
        scores = bm25.score("xyz")
        assert all(s == 0.0 for s in scores)

    def test_term_in_all_docs(self) -> None:
        corpus = ("the cat", "the dog")
        bm25 = BM25Okapi(corpus)
        scores = bm25.score("the")
        # "the" is in both docs — low IDF, but still non-zero
        assert all(s >= 0.0 for s in scores)

    def test_empty_corpus(self) -> None:
        bm25 = BM25Okapi(())
        scores = bm25.score("test")
        assert scores == []

    def test_single_doc(self) -> None:
        bm25 = BM25Okapi(("hello world",))
        scores = bm25.score("hello")
        assert len(scores) == 1
        assert scores[0] > 0.0

    def test_code_tokens(self) -> None:
        corpus = (
            "aiohttp.ClientSession close connection",
            "requests.get url fetch",
            "eval() is dangerous",
        )
        bm25 = BM25Okapi(corpus)

        # Query for "eval" should match doc 2
        scores = bm25.score("eval()")
        assert scores[2] > scores[0]
        assert scores[2] > scores[1]

        # Query for "clientsession" should match doc 0
        scores2 = bm25.score("aiohttp.ClientSession")
        assert scores2[0] > scores2[1]
        assert scores2[0] > scores2[2]


class TestBM25OkapiScoreValues:
    """Verify BM25 score VALUES to kill formula mutation survivors."""

    def test_idf_computation_exact(self) -> None:
        """IDF = log((N - df + 0.5) / (df + 0.5) + 1).

        With 3 docs, a term in 1 doc: IDF = log((3 - 1 + 0.5)/(1 + 0.5) + 1)
                                           = log(2.5/1.5 + 1) = log(2.6667)
        """
        import math

        corpus = ("the cat sat", "the dog ran", "fish swim")
        bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)

        expected_idf_cat = math.log((3 - 1 + 0.5) / (1 + 0.5) + 1.0)
        assert expected_idf_cat == pytest.approx(math.log(2.5 / 1.5 + 1.0))
        # "cat" is in 1 doc — verify through scoring
        scores = bm25.score("cat")
        # doc 0 has "cat" once, doc_len=3, avgdl=7/3≈2.333
        # TF component: tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))
        #             = 1*2.5 / (1 + 1.5*(1-0.75+0.75*3/2.333))
        #             = 2.5 / (1 + 1.5*(0.25+0.75*1.2857))
        #             = 2.5 / (1 + 1.5*(0.25+0.9643))
        #             = 2.5 / (1 + 1.5*1.2143)
        #             = 2.5 / (1 + 1.8214)
        #             = 2.5 / 2.8214
        avgdl = (3 + 3 + 2) / 3  # "the cat sat"=3, "the dog ran"=3, "fish swim"=2
        tf_num = 1 * (1.5 + 1.0)
        tf_den = 1 + 1.5 * (1.0 - 0.75 + 0.75 * 3 / avgdl)
        expected_score_doc0 = expected_idf_cat * tf_num / tf_den

        assert scores[0] == pytest.approx(expected_score_doc0, rel=1e-6)
        assert scores[1] == 0.0  # "cat" not in doc 1
        assert scores[2] == 0.0  # "cat" not in doc 2

    def test_idf_weighting_common_term(self) -> None:
        """Common term ('the') has lower IDF than rare term ('cat')."""
        import math

        corpus = ("the cat sat", "the dog ran", "fish swim")
        bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)

        # "the" appears in 2 of 3 docs
        idf_the = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1.0)
        # "cat" appears in 1 of 3 docs
        idf_cat = math.log((3 - 1 + 0.5) / (1 + 0.5) + 1.0)

        assert idf_cat > idf_the  # rare term has higher IDF

        scores_cat = bm25.score("cat")
        scores_the = bm25.score("the")

        # "cat" score on doc 0 should be higher than "the" score on doc 0
        # because IDF("cat") > IDF("the") even though both appear once in doc 0
        assert scores_cat[0] > scores_the[0]

    def test_score_ordering_three_docs(self) -> None:
        """Query 'cat' scores: doc 0 > doc 1 == doc 2 == 0."""
        corpus = ("the cat sat", "the dog ran", "fish swim")
        bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)
        scores = bm25.score("cat")
        assert scores[0] > 0.0
        assert scores[1] == 0.0
        assert scores[2] == 0.0

    def test_multi_term_query_additive(self) -> None:
        """BM25 score for multi-term query is additive over terms."""
        corpus = ("the cat sat", "the dog ran", "fish swim")
        bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)

        scores_cat = bm25.score("cat")
        scores_sat = bm25.score("sat")
        scores_both = bm25.score("cat sat")

        # "cat sat" = score("cat") + score("sat") for each doc
        for i in range(3):
            assert scores_both[i] == pytest.approx(
                scores_cat[i] + scores_sat[i], rel=1e-6,
            )

    def test_tf_saturation_with_k1(self) -> None:
        """Higher TF increases score but with diminishing returns (k1 saturation)."""
        corpus = ("cat cat cat cat", "cat", "dog")
        bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)
        scores = bm25.score("cat")

        # Doc 0 has TF=4, doc 1 has TF=1
        # Both should be positive, doc 0 higher but NOT 4x higher (saturation)
        assert scores[0] > scores[1] > 0.0
        assert scores[0] < 4.0 * scores[1]  # saturation effect
        assert scores[2] == 0.0

    def test_b_parameter_length_normalization(self) -> None:
        """b=0.75 penalizes longer documents, b=0 ignores length."""
        corpus = ("cat", "cat dog fox bat owl hen")
        # With b=0.75, longer doc is penalized
        bm25_b75 = BM25Okapi(corpus, k1=1.5, b=0.75)
        scores_b75 = bm25_b75.score("cat")
        # Doc 0 (len=1) should score higher than doc 1 (len=6) for same TF=1
        assert scores_b75[0] > scores_b75[1]

        # With b=0, length doesn't matter (only TF and IDF)
        bm25_b0 = BM25Okapi(corpus, k1=1.5, b=0.0)
        scores_b0 = bm25_b0.score("cat")
        # With b=0, same TF=1, same IDF — scores should be equal
        assert scores_b0[0] == pytest.approx(scores_b0[1], rel=1e-6)

    def test_k1_controls_tf_importance(self) -> None:
        """k1=0 makes TF irrelevant (pure IDF); higher k1 increases TF weight."""
        corpus = ("cat cat cat", "cat")
        # k1=0: score is purely IDF * (TF*1 / (TF + 0)) = IDF * 1 for any TF>0
        # Actually k1=0: numerator = tf*(0+1)=tf, denom=tf+0*(...)=tf, so score=IDF
        bm25_k0 = BM25Okapi(corpus, k1=0.0, b=0.75)
        scores_k0 = bm25_k0.score("cat")
        # Both docs have cat, so both get pure IDF (but length norm differs with b>0)
        # With k1=0: num=tf*1=tf, den=tf+0=tf, ratio=1.0 for both
        assert scores_k0[0] == pytest.approx(scores_k0[1], rel=1e-6)

        # k1=10: high TF matters more
        bm25_k10 = BM25Okapi(corpus, k1=10.0, b=0.0)  # b=0 to isolate TF effect
        scores_k10 = bm25_k10.score("cat")
        # Doc 0 (TF=3) should clearly beat doc 1 (TF=1) with high k1
        assert scores_k10[0] > scores_k10[1]

    def test_idf_zero_for_unknown_term(self) -> None:
        """Unknown query term contributes 0 to all scores."""
        corpus = ("cat dog", "fish bird")
        bm25 = BM25Okapi(corpus)
        scores = bm25.score("xyz")
        assert scores == [0.0, 0.0]

    def test_score_strictly_positive_for_match(self) -> None:
        """Any matching doc gets a strictly positive score."""
        corpus = ("alpha beta gamma", "delta epsilon")
        bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)
        scores = bm25.score("alpha")
        assert scores[0] > 0.0
        assert scores[1] == 0.0

    def test_numerator_denominator_formula(self) -> None:
        """Verify numerator = tf*(k1+1), denominator = tf+k1*(1-b+b*dl/avgdl)."""
        import math

        corpus = ("alpha alpha beta",)  # 3 tokens, TF(alpha)=2
        bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)
        scores = bm25.score("alpha")

        # Single doc: avgdl=3, dl=3, N=1, df=1
        # IDF = log((1-1+0.5)/(1+0.5)+1) = log(0.5/1.5+1) = log(1.333)
        idf = math.log((1 - 1 + 0.5) / (1 + 0.5) + 1.0)
        tf = 2
        k1 = 1.5
        b = 0.75
        dl = 3
        avgdl = 3.0
        num = tf * (k1 + 1.0)
        den = tf + k1 * (1.0 - b + b * dl / avgdl)
        expected = idf * num / den
        assert scores[0] == pytest.approx(expected, rel=1e-6)

    def test_inverted_index_tf_accumulation(self) -> None:
        """Term frequency counts multiple occurrences correctly."""
        corpus = ("cat cat cat", "cat")
        bm25 = BM25Okapi(corpus, k1=1.5, b=0.0)
        scores = bm25.score("cat")
        # With b=0: den = tf + k1*(1) = tf + k1
        # doc0: num=3*2.5=7.5, den=3+1.5=4.5, ratio=1.667
        # doc1: num=1*2.5=2.5, den=1+1.5=2.5, ratio=1.0
        # Both have same IDF, so ratio determines ordering
        assert scores[0] / scores[1] == pytest.approx(
            (3 * 2.5 / 4.5) / (1 * 2.5 / 2.5), rel=1e-6,
        )

    def test_avgdl_computation(self) -> None:
        """Average document length affects scores via b parameter."""
        # All same length => dl/avgdl = 1.0 => b term cancels
        corpus_equal = ("ab cd", "ef gh")
        bm25_eq = BM25Okapi(corpus_equal, k1=1.5, b=0.75)

        # Varying lengths => dl/avgdl differs
        corpus_vary = ("ab", "cd ef gh ij")
        bm25_vary = BM25Okapi(corpus_vary, k1=1.5, b=0.75)

        # Query that hits doc 0 in both
        scores_eq = bm25_eq.score("ab")
        scores_vary = bm25_vary.score("ab")

        # corpus_equal: avgdl=2, dl=2, dl/avgdl=1
        # corpus_vary: avgdl=3, dl=1("ab"), dl/avgdl=0.333
        # Shorter doc relative to avg gets boosted with b>0
        assert scores_vary[0] > scores_eq[0]

    def test_default_parameters_k1_b(self) -> None:
        """Default k1=1.5 and b=0.75 produce specific known scores.

        Kills mutants that change default k1=1.5 to 2.5 or b=0.75 to 1.75.
        """
        import math

        corpus = ("cat dog", "cat")
        bm25_default = BM25Okapi(corpus)  # uses defaults k1=1.5, b=0.75

        scores = bm25_default.score("cat")

        # Compute expected with k1=1.5, b=0.75
        # N=2, df(cat)=2, IDF = log((2-2+0.5)/(2+0.5)+1) = log(0.5/2.5+1) = log(1.2)
        idf = math.log((2 - 2 + 0.5) / (2 + 0.5) + 1.0)
        avgdl = (2 + 1) / 2  # 1.5

        # doc 0: tf=1, dl=2
        tf0_num = 1 * (1.5 + 1.0)
        tf0_den = 1 + 1.5 * (1.0 - 0.75 + 0.75 * 2 / avgdl)
        expected_0 = idf * tf0_num / tf0_den

        # doc 1: tf=1, dl=1
        tf1_num = 1 * (1.5 + 1.0)
        tf1_den = 1 + 1.5 * (1.0 - 0.75 + 0.75 * 1 / avgdl)
        expected_1 = idf * tf1_num / tf1_den

        assert scores[0] == pytest.approx(expected_0, rel=1e-6)
        assert scores[1] == pytest.approx(expected_1, rel=1e-6)
        # Shorter doc (doc 1) should score higher with b=0.75
        assert scores[1] > scores[0]

    def test_unknown_term_idf_fallback_zero(self) -> None:
        """Unknown term gets IDF=0 (not 1.0 or None) as fallback.

        Kills mutant that changes idf.get(token, 0.0) to idf.get(token, 1.0).
        """
        corpus = ("cat dog", "fish bird")
        bm25 = BM25Okapi(corpus)
        # "xyz" is unknown — its contribution MUST be 0
        scores_xyz = bm25.score("xyz")
        assert scores_xyz == [0.0, 0.0]

        # "cat xyz" should equal "cat" alone (xyz adds zero)
        scores_cat = bm25.score("cat")
        scores_cat_xyz = bm25.score("cat xyz")
        for i in range(2):
            assert scores_cat_xyz[i] == pytest.approx(scores_cat[i], rel=1e-6)

    def test_idf_zero_skips_term_not_breaks(self) -> None:
        """When IDF=0 (unknown term), 'continue' to next term (not 'break').

        Kills mutant that changes 'continue' to 'break' in score loop.
        Multi-term query: 'xyz cat' — 'xyz' has IDF=0, but 'cat' must still score.
        """
        corpus = ("cat dog", "fish bird")
        bm25 = BM25Okapi(corpus)

        # 'xyz' is unknown (IDF=0), 'cat' is known
        scores = bm25.score("xyz cat")  # xyz first, then cat
        # If 'break', we'd stop after xyz and get [0,0]
        # If 'continue', we skip xyz and score cat
        assert scores[0] > 0.0  # 'cat' must contribute to doc 0

    def test_idf_zero_check_value(self) -> None:
        """The check 'if idf == 0.0' correctly identifies missing terms.

        Kills mutant that changes 'if idf == 0.0' to 'if idf == 1.0'.
        """
        corpus = ("the cat", "the dog")
        bm25 = BM25Okapi(corpus)

        # "the" appears in all docs — IDF is positive but low
        scores_the = bm25.score("the")
        # With 'if idf == 1.0: continue', "the" would NOT be skipped (IDF != 1.0)
        # and would produce scores. This is correct behavior.
        # But mutating to 'idf == 1.0' would also not skip unknown terms,
        # changing behavior for unknown terms.
        # Most direct test: verify known IDF is not 0.0
        import math
        idf_the = math.log((2 - 2 + 0.5) / (2 + 0.5) + 1.0)
        assert idf_the > 0.0  # IDF for "the" is positive
        assert all(s > 0.0 for s in scores_the)  # scores should be positive

        # Unknown term "xyz" should produce all-zero scores
        scores_xyz = bm25.score("xyz")
        assert all(s == 0.0 for s in scores_xyz)


class TestSparseRetrieverEmpty:
    """Edge cases for SparseRetriever."""

    def test_empty_index(self) -> None:
        idx = Index(
            embeddings=np.empty((0, 8), dtype=np.float32),
            rules=(),
            model_name="test",
            dim=8,
            sources={},
        )
        retriever = SparseRetriever()
        result = retriever.retrieve("test", idx, top_k=5, threshold=0.0)
        assert result == []

    def test_none_bm25_corpus(self) -> None:
        rules = _make_rules(2)
        idx = _make_index(rules, bm25_corpus=None)
        retriever = SparseRetriever()
        result = retriever.retrieve("test", idx, top_k=5, threshold=0.0)
        assert result == []


class TestSparseRetrieverBasic:
    """Basic sparse retrieval behavior."""

    def test_returns_scored_candidates(self) -> None:
        rules = _make_rules(3)
        corpus = (
            "never commit secrets to git",
            "use uv for python packages",
            "validate user input at boundaries",
        )
        idx = _make_index(rules, bm25_corpus=corpus)
        retriever = SparseRetriever()
        result = retriever.retrieve("commit secrets", idx, top_k=5, threshold=0.0)

        assert len(result) > 0
        assert all(isinstance(c, ScoredCandidate) for c in result)
        assert all(c.retriever == "sparse" for c in result)
        # "commit secrets" should match first rule best
        assert result[0].rule.text == "Rule 0"

    def test_top_k_limits(self) -> None:
        rules = _make_rules(5)
        corpus = tuple(f"word{i} common term" for i in range(5))
        idx = _make_index(rules, bm25_corpus=corpus)
        retriever = SparseRetriever()
        result = retriever.retrieve("common term", idx, top_k=2, threshold=0.0)
        assert len(result) <= 2

    def test_threshold_filters(self) -> None:
        rules = _make_rules(3)
        corpus = ("alpha beta", "gamma delta", "epsilon zeta")
        idx = _make_index(rules, bm25_corpus=corpus)
        retriever = SparseRetriever()
        result = retriever.retrieve("omega", idx, top_k=5, threshold=0.0)
        # No match, all scores 0.0, threshold is >0 in actual filter (> not >=)
        assert result == []

    def test_query_normalization(self) -> None:
        rules = _make_rules(2)
        corpus = ("commit secrets to git", "validate input")
        idx = _make_index(rules, bm25_corpus=corpus)
        retriever = SparseRetriever()
        # Tool prefix should be stripped
        result = retriever.retrieve("Bash: commit secrets", idx, top_k=5, threshold=0.0)
        assert len(result) > 0
        assert result[0].rule.text == "Rule 0"


class TestSparseParentCollapse:
    """BM25 parent collapse via rule_map."""

    def test_expansion_collapse(self) -> None:
        """With expansions, BM25 collapses to parent rule (max score)."""
        rules = _make_rules(2)
        # Rule 0: canonical + 2 expansions, Rule 1: canonical + 1 expansion
        corpus = (
            "never commit secrets",
            "api keys in git bad",
            "password in source code",
            "use uv for packages",
            "pip alternative uv",
        )
        rule_map = (0, 0, 0, 1, 1)
        idx = _make_index(rules, bm25_corpus=corpus, rule_map=rule_map)

        retriever = SparseRetriever()
        result = retriever.retrieve("api keys git", idx, top_k=5, threshold=0.0)

        # Rule 0 should score highest (expansion "api keys in git bad" matches)
        assert result[0].rule.text == "Rule 0"
        # Only 2 unique parents
        rule_texts = {c.rule.text for c in result}
        assert len(rule_texts) <= 2

    def test_one_result_per_parent(self) -> None:
        """Many expansions for one rule produce exactly one result."""
        rules = _make_rules(1)
        corpus = ("term a", "term b", "term c", "term d")
        rule_map = (0, 0, 0, 0)
        idx = _make_index(rules, bm25_corpus=corpus, rule_map=rule_map)

        retriever = SparseRetriever()
        result = retriever.retrieve("term", idx, top_k=10, threshold=0.0)
        assert len(result) == 1


class TestSparseRetrieverSortOrder:
    """Verify sort order is descending (highest score first).

    Kills mutant that flips np.argsort(-scores) to np.argsort(+scores).
    """

    def test_results_sorted_descending(self) -> None:
        """Results MUST be sorted by score descending (best first)."""
        rules = _make_rules(3)
        # "alpha" appears only in doc 0, "beta" only in doc 1, "gamma" only in doc 2
        corpus = ("alpha unique", "beta unique", "gamma unique")
        idx = _make_index(rules, bm25_corpus=corpus)
        retriever = SparseRetriever()

        # "alpha" matches doc 0 only
        result = retriever.retrieve("alpha beta gamma", idx, top_k=10, threshold=0.0)
        assert len(result) == 3
        scores = [c.score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_highest_scoring_rule_is_first(self) -> None:
        """Rule with unique matching terms is the best result.

        Kills the argsort sign flip mutant directly.
        """
        rules = _make_rules(3)
        # doc 0 has "cat" 3 times, doc 1 has "cat" once, doc 2 has no "cat"
        corpus = ("cat cat cat", "cat dog", "fish bird")
        idx = _make_index(rules, bm25_corpus=corpus)
        retriever = SparseRetriever()

        result = retriever.retrieve("cat", idx, top_k=10, threshold=0.0)
        assert len(result) >= 2
        # doc 0 (TF=3) must be first, doc 1 (TF=1) second
        assert result[0].rule.text == "Rule 0"
        assert result[1].rule.text == "Rule 1"
        assert result[0].score > result[1].score


class TestSparseRetrieverCaching:
    """BM25 index caching behavior.

    Kills mutants in corpus_id tracking and cache invalidation.
    """

    def test_cache_reuses_bm25(self) -> None:
        """Same corpus identity reuses cached BM25 index."""
        rules = _make_rules(2)
        corpus = ("cat dog", "fish bird")
        idx = _make_index(rules, bm25_corpus=corpus)
        retriever = SparseRetriever()

        result1 = retriever.retrieve("cat", idx, top_k=5, threshold=0.0)
        result2 = retriever.retrieve("fish", idx, top_k=5, threshold=0.0)

        # Both should work and return different best results
        assert result1[0].rule.text == "Rule 0"
        assert result2[0].rule.text == "Rule 1"

    def test_cache_invalidates_on_new_corpus(self) -> None:
        """New corpus identity triggers BM25 rebuild."""
        rules = _make_rules(2)
        corpus1 = ("cat dog", "fish bird")
        idx1 = _make_index(rules, bm25_corpus=corpus1)

        corpus2 = ("fish bird", "cat dog")  # same content, different tuple identity
        idx2 = _make_index(rules, bm25_corpus=corpus2)

        retriever = SparseRetriever()
        result1 = retriever.retrieve("cat", idx1, top_k=5, threshold=0.0)
        assert result1[0].rule.text == "Rule 0"  # "cat dog" is first

        result2 = retriever.retrieve("cat", idx2, top_k=5, threshold=0.0)
        assert result2[0].rule.text == "Rule 1"  # "cat dog" is second now
