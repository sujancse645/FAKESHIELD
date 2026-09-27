"""claim_extractor.py - Deterministic, lightweight factual claim extractor for FakeShield.

Extracts checkable factual claims from English text or Phase 6 English pivot representation.
Filters opinions, rhetorical questions, and vague commentary.
Assigns claim types: policy, event, statistic, quote, entity_fact, other.
Limits extraction to maximum 5 claims per input.
"""

import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


CLAIM_TYPES = ["policy", "event", "statistic", "quote", "entity_fact", "other"]

# Patterns for filtering opinions, subjective commentary, and non-factual statements
OPINION_PATTERNS = [
    r"\b(?:i\s+think|i\s+believe|in\s+my\s+opinion|in\s+my\s+view|i\s+feel)\b",
    r"\b(?:we\s+hope|we\s+wish|it\s+seems\s+to\s+me|as\s+far\s+as\s+i\s+know)\b",
    r"\b(?:wonderful|terrible|disgusting|horrible|glorious|magnificent|awesome)\b",
    r"\b(?:should\s+be\s+ashamed|deserves\s+praise|shameful|unacceptable)\b",
    r"\b(?:wake\s+up\s+people|let\s+us\s+unite|forward\s+this|share\s+this)\b",
]

# Patterns for question and rhetorical sentence filtering
QUESTION_PATTERNS = [
    r"\?+\s*$",
    r"^(?:who|what|where|when|why|how|which|whose|whom)\b",
    r"^(?:is\s+it\s+true|can\s+anyone\s+explain|do\s+you\s+know|did\s+you\s+hear)\b",
    r"^(?:could\s+this\s+be|will\s+they\s+ever|should\s+we\s+trust)\b",
]

# Patterns for claim type identification
STATISTIC_PATTERNS = [
    r"\b\d+(?:[\.,]\d+)?\s*(?:%|percent|crore|lakh|million|billion|trillion|rupees|rs|usd|\$|euro|km|kilometers|kg|tons|cusecs|hours|days|years)\b",
    r"\b\d{1,3}(?:,\d{3})+\b",
    r"\b(?:increased|decreased|fell|rose|dropped|surged|jumped|reduced|doubled|tripled)\s+by\s+\d+",
    r"\b(?:rate|inflation|gdp|deficit|growth|unemployment|interest)\s+(?:at|of|reached)\s+\d+",
]

POLICY_PATTERNS = [
    r"\b(?:government|cabinet|ministry|parliament|assembly|rbi|reserve\s+bank|supreme\s+court|high\s+court|commission|authority)\b.*?\b(?:announced|approved|launched|enacted|ordered|ruled|introduced|decided|cleared|banned|allocated|declared)\b",
    r"\b(?:policy|scheme|ordinance|bill|act|resolution|guideline|directive|subsidy|quota|pension|welfare)\b",
    r"\b(?:repo\s+rate|interest\s+rate|monetary\s+policy|fiscal\s+deficit)\b",
]

QUOTE_PATTERNS = [
    r"[\"“][^\"”]{10,}[\"”]",
    r"\b(?:said|stated|remarked|claimed|alleged|confirmed|denied|told\s+reporters|quoted\s+as\s+saying)\b",
]

EVENT_PATTERNS = [
    r"\b(?:isro|nasa|police|military|army|navy|air\s+force|coast\s+guard)\b.*?\b(?:launched|arrested|rescued|intercepted|deployed|conducted|tested)\b",
    r"\b(?:cyclone|earthquake|flood|landslide|fire|accident|crash|blast|explosion|storm)\b.*?\b(?:hit|struck|occurred|damaged|killed|injured|disrupted)\b",
    r"\b(?:inaugurated|opened|completed|signed|held|conducted|organized|celebrated)\b",
]

ENTITY_FACT_PATTERNS = [
    r"\b(?:is\s+the\s+capital|is\s+the\s+largest|is\s+the\s+oldest|is\s+the\s+headquarters|was\s+founded\s+in|headquartered\s+in)\b",
    r"\b(?:located\s+in|borders|measures|stretches\s+across)\b",
]


@dataclass
class ExtractedClaim:
    """Individual factual checkable statement."""
    claim_id: str
    claim: str
    type: str
    source_sentence: str
    search_query: str
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimSet:
    """Structured set of extracted claims for a single text input."""
    input_id: str
    raw_text: str
    english_text: str
    source_language: str
    claims: List[ExtractedClaim]
    status: str
    total_claims: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_id": self.input_id,
            "raw_text": self.raw_text,
            "english_text": self.english_text,
            "source_language": self.source_language,
            "claims": [c.to_dict() for c in self.claims],
            "status": self.status,
            "total_claims": self.total_claims,
        }


class ClaimExtractor:
    """Extracts factual, checkable claims from English text."""

    def __init__(self, max_claims: int = 5):
        self.max_claims = max_claims

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences while protecting common abbreviations."""
        if not text:
            return []

        cleaned = unicodedata.normalize("NFKC", str(text))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Protect common abbreviations and numbers
        protected = cleaned
        abbrevs = [
            ("Mr.", "Mr_DOT_"), ("Mrs.", "Mrs_DOT_"), ("Dr.", "Dr_DOT_"),
            ("Prof.", "Prof_DOT_"), ("Gov.", "Gov_DOT_"), ("U.S.", "U_DOT_S_DOT_"),
            ("R.B.I.", "R_DOT_B_DOT_I_DOT_"), ("I.S.R.O.", "I_DOT_S_DOT_R_DOT_O_DOT_"),
            ("No.", "No_DOT_"), ("vs.", "vs_DOT_"), ("approx.", "approx_DOT_"),
            ("Jan.", "Jan_DOT_"), ("Feb.", "Feb_DOT_"), ("Oct.", "Oct_DOT_"), ("Nov.", "Nov_DOT_"),
        ]
        for src, repl in abbrevs:
            protected = protected.replace(src, repl)

        # Protect decimals like 6.5
        protected = re.sub(r"(\d+)\.(\d+)", r"\1_DEC_\2", protected)

        # Split on sentence boundaries (. ! ?)
        raw_sents = re.split(r"(?<=[.!?])\s+", protected)

        results = []
        for s in raw_sents:
            # Restore protected patterns
            for src, repl in abbrevs:
                s = s.replace(repl, src)
            s = s.replace("_DEC_", ".")
            s = s.strip()
            if s:
                results.append(s)

        return results

    def is_question(self, sentence: str) -> bool:
        """Check if sentence is a question or rhetorical inquiry."""
        s = sentence.strip().lower()
        for p in QUESTION_PATTERNS:
            if re.search(p, s, flags=re.IGNORECASE):
                return True
        return False

    def is_opinion(self, sentence: str) -> bool:
        """Check if sentence is subjective, emotional, or pure commentary."""
        s = sentence.strip().lower()
        for p in OPINION_PATTERNS:
            if re.search(p, s, flags=re.IGNORECASE):
                return True
        return False

    def classify_claim_type(self, sentence: str) -> str:
        """Categorize a factual claim into one of the designated types."""
        s = sentence.lower()

        # Check statistic first (quantitative specificity)
        for p in STATISTIC_PATTERNS:
            if re.search(p, s, flags=re.IGNORECASE):
                return "statistic"

        # Check policy statements
        for p in POLICY_PATTERNS:
            if re.search(p, s, flags=re.IGNORECASE):
                return "policy"

        # Check quotations
        for p in QUOTE_PATTERNS:
            if re.search(p, sentence):
                return "quote"

        # Check event statements
        for p in EVENT_PATTERNS:
            if re.search(p, s, flags=re.IGNORECASE):
                return "event"

        # Check entity facts
        for p in ENTITY_FACT_PATTERNS:
            if re.search(p, s, flags=re.IGNORECASE):
                return "entity_fact"

        return "other"

    def normalize_claim(self, sentence: str, prior_subject: Optional[str] = None) -> str:
        """Clean sentence and resolve simple leading anaphora if prior subject is available."""
        s = sentence.strip()
        # Remove surrounding quotation marks if they enclose the whole sentence
        if (s.startswith('"') and s.endswith('"')) or (s.startswith('“') and s.endswith('”')):
            s = s[1:-1].strip()

        # Simple anaphora resolution
        if prior_subject:
            match = re.match(r"^(He|She|It|They)\s+(announced|stated|said|confirmed|launched|decided)\b", s, flags=re.IGNORECASE)
            if match:
                pronoun = match.group(1)
                verb = match.group(2)
                rest = s[len(pronoun) + len(verb) + 1:].strip()
                s = f"{prior_subject} {verb} {rest}"

        return s

    def _extract_subject_entity(self, sentence: str) -> Optional[str]:
        """Extract primary leading entity to serve as antecedent for subsequent sentences."""
        # Check capitalized proper nouns near beginning
        m = re.match(r"^((?:The\s+)?(?:[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*))\b", sentence)
        if m:
            candidate = m.group(1).strip()
            # Avoid generic words like 'However', 'Therefore', 'Yesterday'
            if candidate not in ("However", "Therefore", "Yesterday", "Today", "Furthermore", "Meanwhile"):
                return candidate
        return None

    def generate_search_query(self, claim_text: str) -> str:
        """Generate a concise 4-8 keyword search query from claim text."""
        # Remove punctuation except hyphens
        cleaned = re.sub(r"[^\w\s-]", " ", claim_text)
        tokens = cleaned.split()

        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "to", "from", "in", "on",
            "at", "by", "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below", "of", "and",
            "or", "but", "if", "while", "that", "this", "these", "those", "then",
            "so", "than", "too", "very", "can", "will", "just", "should", "now"
        }

        # Select important tokens: numbers, capitalized terms, or non-stopwords
        selected = []
        for t in tokens:
            if re.search(r"\d", t) or t[0].isupper() or t.lower() not in stopwords:
                selected.append(t)

        # Limit to 7 most relevant tokens
        if len(selected) > 7:
            selected = selected[:7]

        query = " ".join(selected).strip()
        return query if query else claim_text[:50]

    def extract_claims(
        self,
        text: str,
        input_id: str = "input_01",
        source_language: str = "en"
    ) -> ClaimSet:
        """Extract factual claims from input English text."""
        if not text or len(text.strip()) < 5:
            return ClaimSet(
                input_id=input_id,
                raw_text=text or "",
                english_text=text or "",
                source_language=source_language,
                claims=[],
                status="insufficient_text",
                total_claims=0
            )

        sentences = self._split_sentences(text)
        candidates = []
        prior_subject = None

        for s in sentences:
            # 1. Skip questions
            if self.is_question(s):
                continue

            # 2. Skip subjective opinions
            if self.is_opinion(s):
                continue

            # 3. Skip extreme short fragments (< 4 words)
            words = s.split()
            if len(words) < 4:
                continue

            # Normalize claim and resolve anaphora if possible
            norm_claim = self.normalize_claim(s, prior_subject=prior_subject)

            # Update prior subject for next iteration
            subj = self._extract_subject_entity(s)
            if subj:
                prior_subject = subj

            # Classify type
            ctype = self.classify_claim_type(norm_claim)

            # Generate search query
            query = self.generate_search_query(norm_claim)

            candidates.append({
                "claim": norm_claim,
                "type": ctype,
                "source_sentence": s,
                "search_query": query,
            })

        # Order by centrality/factual weight: statistics and policies first, then events/quotes
        type_priority = {"statistic": 0, "policy": 1, "event": 2, "quote": 3, "entity_fact": 4, "other": 5}
        candidates.sort(key=lambda x: type_priority.get(x["type"], 99))

        # Enforce maximum 5 claims limit
        selected_candidates = candidates[:self.max_claims]

        extracted = []
        for i, c in enumerate(selected_candidates, start=1):
            claim_obj = ExtractedClaim(
                claim_id=f"{input_id}_c{i:02d}",
                claim=c["claim"],
                type=c["type"],
                source_sentence=c["source_sentence"],
                search_query=c["search_query"],
                confidence=None  # Explicitly None to prevent confusion with truth score
            )
            extracted.append(claim_obj)

        status = "success" if extracted else "no_claims_found"

        return ClaimSet(
            input_id=input_id,
            raw_text=text,
            english_text=text,
            source_language=source_language,
            claims=extracted,
            status=status,
            total_claims=len(extracted)
        )
