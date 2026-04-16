COMMUNICATION_ANALYSIS_PROMPT = """
You are a fraud analyst. Analyze messages for scams and return STRICT JSON with keys:
scam_probability (0-1), pressure_tactics (list), impersonation_signals (list),
money_movement_signals (list), urgency_score (0-1), reasoning_summary (short string).
Input:\n{summary}
""".strip()

PATTERN_SUMMARY_PROMPT = """
Summarize emerging fraud patterns from these high-risk clusters.
Focus on: temporal pattern, network behavior, geo inconsistency, social engineering cues.
Return compact "pattern cards" with names and when they trigger.
Data:\n{cluster_dump}
""".strip()

ARBITRATION_PROMPT = """
Given borderline fraud cases with features and agent evidence, return a concise ranking rationale.
Cases:\n{cases}
""".strip()
