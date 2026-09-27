"""query_builder.py - Search query generation for FakeShield evidence retrieval.

Generates concise, focused 4-8 keyword search queries from extracted claims.
Preserves core named entities, dates, numeric quantities, and key actions.
Removes grammatical stopwords and conversational fillers.
"""

import re
from typing import List, Optional


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "to", "from", "in", "on",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "of", "and",
    "or", "but", "if", "while", "that", "this", "these", "those", "then",
    "so", "than", "too", "very", "can", "will", "just", "should", "now",
    "would", "could", "also", "such", "there", "their", "its", "our", "all"
}


class QueryBuilder:
    """Builds concise English search queries optimized for fact-check search."""

    def __init__(self, max_keywords: int = 7):
        self.max_keywords = max_keywords

    def build_query(self, claim_text: str) -> str:
        """Generate a concise search query from claim text.

        Preserves:
          - Named entities (capitalized terms, acronyms like RBI, ISRO, WHO)
          - Quantitative numbers (including decimals e.g. 6.5, currency, percentages)
          - Action verbs (e.g. approved, announced, launched, increased)
        """
        if not claim_text:
            return ""

        # Protect decimals and percentages
        text = claim_text.strip()

        # Tokenize by whitespace while preserving punctuation attached to numbers
        raw_tokens = text.split()
        selected_tokens = []

        for raw_tok in raw_tokens:
            # Strip trailing commas or periods that are punctuation, but preserve internal decimals (6.5)
            tok = raw_tok.strip(".,;:\"'!?()[]{}")
            if not tok:
                continue

            # 1. Always retain numbers and percentages (e.g., '6.5', '50000', '15', '2026', '6.5%')
            if re.search(r"\d", tok):
                selected_tokens.append(tok)
                continue

            # 2. Always retain capitalized named entities and acronyms (e.g., 'RBI', 'ISRO', 'Chennai')
            if tok[0].isupper() and tok.lower() not in STOPWORDS:
                selected_tokens.append(tok)
                continue

            # 3. Retain key non-stopword tokens (e.g., 'crop', 'loans', 'warning', 'repo', 'satellite')
            if tok.lower() not in STOPWORDS and len(tok) > 2:
                selected_tokens.append(tok)

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for t in selected_tokens:
            lower_t = t.lower()
            if lower_t not in seen:
                seen.add(lower_t)
                deduped.append(t)

        # Restrict to max_keywords for a concise query
        if len(deduped) > self.max_keywords:
            deduped = deduped[:self.max_keywords]

        query = " ".join(deduped).strip()
        return query if query else claim_text[:60]
