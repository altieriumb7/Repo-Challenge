COMMUNICATION_ANALYSIS_PROMPT = """
You are a fraud analyst. Given summarized communications, list scam signals in bullets.
Input:\n{summary}
""".strip()

PATTERN_SUMMARY_PROMPT = """
Summarize emerging fraud patterns from these high-risk clusters.
Focus on: temporal pattern, network behavior, geo inconsistency, social engineering cues.
Data:\n{cluster_dump}
""".strip()

ARBITRATION_PROMPT = """
Given borderline fraud cases with features and agent evidence, return a concise ranking rationale.
Cases:\n{cases}
""".strip()
