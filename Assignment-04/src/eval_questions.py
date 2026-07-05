"""
Canonical list of the 10 fixed IT Helpdesk evaluation questions used across
all three notebooks (base model baseline, SFT comparison, DPO final
evaluation) and the corresponding report tables in `reports/`.

Keeping this list in one place ensures the base/SFT/DPO answers all line up
row-for-row when `fill_reports.py` assembles the comparison tables.
"""

EVAL_QUESTIONS = [
    "Unable to Access Office Applications after password reset.",
    "My VPN keeps disconnecting every few minutes, how do I fix it?",
    "The reporting dashboard is not loading any widgets.",
    "I was charged extra fees on my invoice this month, why?",
    "How do I set up the integration with our third-party CRM tool?",
    "Our production system just went down, what should I do?",
    "I forgot my password and the reset link is not arriving.",
    "Why is our Wi-Fi so slow today across the whole office?",
    "Can you resend last month's invoice?",
    "The bulk data import keeps failing with validation errors.",
]
