"""source_reliability.py - Source Reliability Tiering for FakeShield.

Assigns transparent reliability tiers (1 to 4) to evidence sources based on
explainable rules defined in configs/source_reliability.yaml.

CRITICAL PRINCIPLE:
Source tier is NOT truth value.
- Tier 1 sources can be mistaken or irrelevant to a specific claim.
- Tier 4 sources are not automatically false.
- Tiering is an objective provenance feature used in subsequent evidence fusion.
"""

import os
import re
from urllib.parse import urlparse
from typing import Dict, Any, Optional
import yaml


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs",
    "source_reliability.yaml"
)


class SourceReliabilityAnalyzer:
    """Evaluates the reliability tier of evidence sources based on configurable rules."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load source reliability configuration from YAML."""
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        # Fallback default configuration
        return {
            "tiers": {
                1: {"name": "Official Primary Source", "domains": [".gov", ".gov.in", ".nic.in", "rbi.org.in", "who.int", "un.org"]},
                2: {"name": "Established Fact-Checking Organization", "domains": ["altnews.in", "boomlive.in", "snopes.com", "politifact.com"]},
                3: {"name": "Reputable News Organization", "domains": ["thehindu.com", "reuters.com", "bbc.com", "indianexpress.com"]},
                4: {"name": "Unknown / Unverified Source", "domains": ["twitter.com", "facebook.com", "youtube.com"]}
            },
            "default_tier": 4,
            "default_tier_name": "Unknown / Unverified Source"
        }

    @staticmethod
    def extract_domain(url_or_domain: str) -> str:
        """Extract clean canonical domain from URL or domain string."""
        if not url_or_domain:
            return ""

        s = str(url_or_domain).strip().lower()
        if not s.startswith("http://") and not s.startswith("https://"):
            s = "https://" + s

        try:
            parsed = urlparse(s)
            domain = parsed.netloc or parsed.path
            # Strip port if present
            domain = domain.split(":")[0]
            # Strip leading www.
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return url_or_domain.lower().strip()

    def get_tier(self, url_or_domain: str) -> Dict[str, Any]:
        """Determine the reliability tier of a given URL or domain.

        Returns:
            Dict with source, domain, tier (1-4), tier_name, rule, and description.
        """
        domain = self.extract_domain(url_or_domain)
        full_url = str(url_or_domain).lower()

        tiers_cfg = self.config.get("tiers", {})

        s = str(url_or_domain).strip().lower()
        if not s.startswith("http://") and not s.startswith("https://"):
            s = "https://" + s
        try:
            parsed = urlparse(s)
            path = parsed.path.lower()
        except Exception:
            path = ""

        def _match_pattern(pat: str) -> bool:
            if pat.startswith("."):
                return domain.endswith(pat) or f".{domain}".endswith(pat)
            if "/" in pat:
                pat_dom, pat_path = pat.split("/", 1)
                dom_match = (domain == pat_dom or domain.endswith(f".{pat_dom}"))
                path_match = path.lstrip("/").startswith(pat_path.lstrip("/"))
                return dom_match and path_match
            return domain == pat or domain.endswith(f".{pat}")

        # Check Tiers in ascending order of authority: 1, 2, 3
        for tier_num in [1, 2, 3]:
            tier_data = tiers_cfg.get(tier_num, {})
            name = tier_data.get("name", f"Tier {tier_num}")
            desc = tier_data.get("description", "")
            domains = tier_data.get("domains", [])

            for pattern in domains:
                pat = pattern.lower().strip()
                if _match_pattern(pat):
                    return {
                        "source": url_or_domain,
                        "domain": domain,
                        "tier": tier_num,
                        "tier_name": name,
                        "rule": f"Matched pattern '{pat}'",
                        "description": desc,
                    }

        # Check Tier 4 known patterns
        tier_4_data = tiers_cfg.get(4, {})
        for pattern in tier_4_data.get("domains", []):
            pat = pattern.lower().strip()
            if _match_pattern(pat):
                return {
                    "source": url_or_domain,
                    "domain": domain,
                    "tier": 4,
                    "tier_name": tier_4_data.get("name", "Unknown / Unverified Source"),
                    "rule": f"Matched known unverified/social pattern '{pat}'",
                    "description": tier_4_data.get("description", ""),
                }

        # Default fallback to Tier 4
        return {
            "source": url_or_domain,
            "domain": domain,
            "tier": self.config.get("default_tier", 4),
            "tier_name": self.config.get("default_tier_name", "Unknown / Unverified Source"),
            "rule": "Domain not in recognized registry (default tier)",
            "description": "Unindexed domain, independent blog, or unverified publisher",
        }
