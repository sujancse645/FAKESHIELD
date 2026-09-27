# FakeShield — Project Specification

## Project Title

FakeShield: Evidence-Grounded Multilingual Multimodal News Verification System

---

# 1. Project Overview

FakeShield is a machine-learning-based news verification system designed to analyze potentially misleading or false information appearing as text, newspaper clippings, screenshots, and URLs.

Unlike a conventional fake-news classifier that attempts to predict whether text "looks fake", FakeShield follows an evidence-grounded verification approach.

The system extracts factual claims, retrieves relevant evidence, evaluates the relationship between the claim and evidence, considers source reliability, and produces an explainable verdict.

The system is designed especially for multilingual Indian news and image-based misinformation.

---

# 2. Problem Statement

Existing fake-news classification systems can achieve high performance on benchmark datasets while learning dataset-specific patterns such as writing style, vocabulary, publisher identity, or formatting rather than factual truth.

This creates a major generalization problem when the system encounters news from an unseen publisher, language, format, or real-world source.

FakeShield addresses this problem by moving from simple text classification toward evidence-grounded claim verification.

The proposed system combines:

1. Multimodal input processing
2. OCR for image-based news
3. Language detection
4. English-pivot multilingual translation
5. Claim extraction
6. Evidence retrieval
7. Source reliability analysis
8. Natural Language Inference (NLI)
9. Evidence fusion
10. Uncertainty-aware abstention
11. Explainable verdict generation

---

# 3. Core Principle

FakeShield must NOT claim that an ML model can directly determine absolute truth.

The system should instead answer:

"Based on the available evidence, does the evidence support, contradict, or fail to verify this claim?"

Therefore, the system must be capable of saying:

"I don't have enough reliable evidence to determine this."

This uncertainty-aware behavior is a core design principle.

---

# 4. Target Users

The initial target users are:

- Students
- General internet users
- Social-media users
- People receiving forwarded news
- Users reading newspaper screenshots
- Users who encounter potentially misleading information in Indian languages

The system is intended as an information-verification assistant and not as an authoritative legal, governmental, medical, or financial decision-maker.

---

# 5. Supported Input Types

FakeShield will initially support four input types.

## 5.1 Plain Text

The user can paste news or a factual claim.

Example:

"Government announces ₹50,000 assistance for every engineering student."

---

## 5.2 Newspaper / News Image

The user can upload:

- Newspaper clipping
- Photograph of newspaper
- Printed article
- News poster
- Screenshot

The system will perform OCR before verification.

---

## 5.3 Social Media / WhatsApp Screenshot

The user can upload screenshots containing:

- Forwarded messages
- News cards
- Social-media posts
- Text embedded in images

The system will extract the text and attempt verification.

---

## 5.4 URL

The user can provide a news/article URL.

The system may extract the article text and pass it through the same verification pipeline.

URL processing will be implemented only after the core text/image pipeline is stable.

---

# 6. Supported Languages

## Initial Prototype

The first prototype will support:

1. English
2. Tamil
3. Hindi

These languages are selected to establish:

- English baseline capability
- South Indian language capability
- North Indian language capability

---

## Future Language Expansion

The architecture should allow future support for:

- Telugu
- Malayalam
- Kannada
- Bengali
- Marathi
- Gujarati
- Punjabi
- Other Indian languages

The project must NOT claim perfect support for all 22 scheduled Indian languages during the initial prototype.

The architecture may be designed for multilingual expansion, but experimental claims must only be made for languages actually tested.

---

# 7. Proposed Architecture

The initial system architecture is:

INPUT
↓
Input Type Detection
↓
Image Preprocessing (if image)
↓
OCR (if image)
↓
Language Detection
↓
Translation to English (if required)
↓
Claim Extraction
↓
Evidence Retrieval
↓
Source Reliability Analysis
↓
NLI-based Evidence Evaluation
↓
Evidence Fusion
↓
Abstention / Uncertainty Check
↓
Final Verdict
↓
Explanation
↓
Translation of Explanation
↓
User Interface

---

# 8. Multilingual Strategy

FakeShield will initially use English as a pivot language.

For Indian-language input:

Native-language text
↓
Language Detection
↓
English Translation
↓
Evidence Retrieval
↓
NLI Verification
↓
Verdict
↓
Translation back to original language

The original-language text must always be preserved.

The system must never silently replace the original text with the translated text.

The interface should allow users to compare:

- Original text
- English translation
- Verification result

---

# 9. Image Verification Strategy

FakeShield will not initially attempt to solve the entire problem of deepfake image detection.

The first image-verification objective is:

Image
↓
OCR
↓
Extracted Text
↓
Claim Verification

The system will focus on text-bearing misinformation such as:

- Newspaper clippings
- Screenshots
- Posters
- Forwarded messages

Visual manipulation/deepfake detection may be considered future work.

---

# 10. Claim-Level Verification

FakeShield should not blindly classify an entire article as fake or real.

The system should extract individual factual claims.

Example:

Input:

"The Tamil Nadu government announced ₹50,000 assistance for every engineering student. The scheme will begin in October."

Claims:

1. Tamil Nadu government announced ₹50,000 assistance.
2. The assistance applies to every engineering student.
3. The scheme will begin in October.

Each claim should be independently evaluated when possible.

---

# 11. Evidence Retrieval

Evidence should be retrieved from external fact-checking and reliable information sources.

The initial prototype will investigate:

- Google Fact Check Tools / ClaimReview data
- Official government sources
- Established fact-checking organizations
- Reputable news organizations

The system should preserve:

- Evidence text
- URL
- Publisher
- Publication/review date
- Fact-check rating where available
- Source tier

---

# 12. Source Reliability

Source reliability will initially be implemented as a transparent rule-based layer.

Sources may be assigned tiers based on predefined criteria.

Example conceptual structure:

Tier 1:
Official government / institutional sources

Tier 2:
Established fact-checking organizations

Tier 3:
Established news organizations

Tier 4:
Unknown or unclassified sources

Source tier must be treated as a reliability prior and NOT as proof that an individual article is true.

The source-tier configuration must be documented and version-controlled.

---

# 13. Natural Language Inference

Retrieved evidence will be evaluated against each extracted claim using a pretrained multilingual NLI model.

The expected NLI relationships are:

1. Entailment
2. Contradiction
3. Neutral

Conceptually:

Evidence → Claim

Entailment:
Evidence supports the claim.

Contradiction:
Evidence conflicts with the claim.

Neutral:
Evidence does not establish whether the claim is true or false.

NLI output must not be treated as absolute truth.

---

# 14. Final Verdict Classes

FakeShield will use five primary verdict categories.

## VERIFIED

Available reliable evidence supports the claim.

## CONTRADICTED

Available reliable evidence contradicts the claim.

## MISLEADING

Evidence suggests that the claim contains a mixture of accurate and inaccurate information or lacks important context.

## OUT-OF-CONTEXT

The underlying information may be genuine, but the evidence indicates that it is old, reused, or presented as current when it is not.

## UNVERIFIED

There is insufficient reliable evidence to confidently support or contradict the claim.

---

# 15. Abstention Principle

FakeShield must not force a verdict when evidence is insufficient.

If:

- Evidence is weak
- Evidence is irrelevant
- NLI is mostly neutral
- Sources are unreliable
- Retrieval returns nothing
- Confidence is below the defined threshold

the system should prefer:

UNVERIFIED

rather than incorrectly predicting:

VERIFIED or CONTRADICTED.

This behavior will be experimentally evaluated.

---

# 16. Explainability Requirement

Every final verdict should provide an explanation.

The explanation should contain, where available:

- Extracted claim
- Evidence supporting the claim
- Evidence contradicting the claim
- Source information
- NLI relationship
- Reason for the final verdict
- Confidence/verification score
- Uncertainty information

The system should not display only:

"FAKE — 95%"

without an explanation.

---

# 17. Confidence Terminology

The project must avoid presenting a model score as an absolute probability of truth.

The preferred terminology is:

"Verification Confidence"

or

"Evidence Confidence"

The score represents confidence in the system's verification decision based on available signals.

It does NOT represent an absolute probability that the claim is objectively true.

---

# 18. Research Questions

The project will investigate the following research questions.

## RQ1

Does evidence-grounded verification improve performance compared with text-only fake-news classification?

## RQ2

How does OCR affect end-to-end news verification performance?

## RQ3

How effective is English-pivot verification for Indian-language claims?

## RQ4

Can uncertainty-aware abstention reduce false-positive and false-negative decisions?

## RQ5

How does FakeShield perform on real-world newspaper clippings and social-media screenshots?

---

# 19. Baseline System

A conventional ML baseline will be implemented for comparison.

Baseline:

TF-IDF
↓
Logistic Regression

Additional classical models may include:

- Naive Bayes
- Random Forest

The baseline will be evaluated separately from the evidence-grounded system.

The project must not rely only on random train/test accuracy.

Where possible, cross-source or independent evaluation should be used to investigate dataset/source leakage.

---

# 20. Proposed System

The proposed FakeShield system is:

Multilingual/Multimodal Input
↓
OCR
↓
Language Detection
↓
Translation
↓
Claim Extraction
↓
Evidence Retrieval
↓
Source Reliability
↓
NLI
↓
Evidence Fusion
↓
Abstention
↓
Explainable Verdict

The proposed system will be compared with the baseline.

---

# 21. Evaluation Metrics

The project will evaluate different components independently.

## ML Classification

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## OCR

- Character Error Rate (CER)
- Word Error Rate (WER)

## Retrieval

- Relevant evidence retrieval rate
- Retrieval success rate
- Manual relevance assessment

## NLI

- Accuracy
- Precision
- Recall
- F1-score
- Human agreement

## End-to-End System

- Correct verdict rate
- False-positive rate
- False-negative rate
- Abstention rate
- Coverage
- Accuracy at different thresholds

## Annotation

- Cohen's Kappa

---

# 22. Initial Golden Set

Before system development, a 20-item golden set must be created.

The golden set will contain:

- 5 Tamil newspaper clippings
- 5 Tamil/Hindi WhatsApp or social-media screenshots
- 5 English factual claims
- 5 genuine news items

Each item should have:

- Unique ID
- Input type
- Language
- Original input
- Ground-truth label
- Ground-truth evidence URL
- Notes

The golden set will be used as a regression test throughout development.

---

# 23. Golden Set Rule

The golden set must NOT be repeatedly modified simply to make the system look better.

It should remain stable during development.

Every major change must be tested against the same 20 items.

Results must be recorded.

If an improvement causes another previously working case to fail, the regression must be documented.

---

# 24. Manual Verification Principle

No development phase may be considered complete merely because the program executes without an error.

Every phase requires:

1. Automated test
2. Manual inspection
3. Error logging
4. Pass/fail decision

A phase may proceed only after its manual gate is passed.

---

# 25. Research Integrity

The project must never:

- Invent evidence
- Invent citations
- Fabricate evaluation results
- Claim unsupported accuracy
- Treat model confidence as truth probability
- Remove failed test cases simply to improve results
- Modify ground truth to match predictions

All reported results must come from actual experiments.

Known failures must be included in the final report.

---

# 26. Known Limitations

The initial system may have limitations involving:

- OCR errors
- Low-quality images
- Complex newspaper layouts
- Translation errors
- Low-resource languages
- Missing fact-check coverage
- Outdated evidence
- NLI errors
- Ambiguous claims
- Satire
- Opinion articles
- Breaking news where evidence does not yet exist

These limitations will be recorded rather than hidden.

---

# 27. Scope for Initial Semester Project

The minimum successful implementation is:

1. English text verification
2. Tamil image verification
3. Hindi text/image verification
4. OCR
5. English-pivot translation
6. Claim extraction
7. Fact-check retrieval
8. NLI
9. Evidence-based verdict
10. Abstention
11. Explainable result
12. Streamlit demonstration

Advanced features such as additional languages, URL extraction, image similarity and sophisticated multimodal models are secondary.

---

# 28. Definition of Success

FakeShield is considered technically successful when:

A user can provide a supported text or image input,

and the system can:

1. Extract the text
2. Detect the language
3. Translate it when necessary
4. Extract at least one factual claim
5. Retrieve evidence when available
6. Evaluate evidence against the claim
7. Produce an appropriate verdict
8. Explain the verdict
9. Abstain when evidence is insufficient

The system does not need to correctly verify every claim on the internet.

A scientifically honest "UNVERIFIED" result is considered a valid system output.

---

# 29. Development Philosophy

Development must follow an end-to-end-first approach.

The team must build a complete but imperfect pipeline on the 20-item golden set before spending significant time on optimization.

The priority order is:

1. Working pipeline
2. Manual verification
3. Measurement
4. Error analysis
5. Optimization
6. Research experiments
7. UI polish

The team must not spend several weeks optimizing one component while the remaining pipeline is disconnected.

---

# 30. Phase Completion Rule

Every development phase must produce:

1. Code
2. Test output
3. Manual verification
4. Error log
5. Evidence of completion

Only then can the next phase begin.

---

# 31. Final Project Goal

FakeShield aims to demonstrate that reliable misinformation assistance should not simply ask:

"Does this text look fake?"

Instead, it should ask:

"What factual claim is being made, what evidence exists, what does that evidence imply, how reliable is the evidence, and is there enough information to make a responsible decision?"

The final system therefore follows:

SCAN → UNDERSTAND → VERIFY → EXPLAIN


PHASE 0 VERIFICATION

[ ] Project title finalized
[ ] Problem statement finalized
[ ] Inputs finalized
[ ] Outputs finalized
[ ] Initial languages finalized
[ ] Architecture finalized
[ ] Research questions finalized
[ ] Golden-set concept understood
[ ] Evidence-grounded approach understood
[ ] NLI concept understood
[ ] Abstention concept understood
[ ] Baseline vs proposed system understood
[ ] Both members reviewed PROJECT_SPEC.md
[ ] Both members can explain the complete pipeline
[ ] No coding started before specification freeze

PHASE 0 STATUS: __________

Member A: __________
Member B: __________
Date: __________
