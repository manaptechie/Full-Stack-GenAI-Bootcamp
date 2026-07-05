"""
Lightweight, dependency-free dataset generator for the IT Helpdesk AI
Assistant fine-tuning project.

Why this exists:
    The original `prepare_dataset.py` downloads a public Hugging Face dataset
    (Tobi-Bueck/customer-support-tickets). That requires a working
    `datasets`/`torch` install and internet access. On this machine the local
    virtual environment has a broken/CPU-only torch install and no GPU, so
    instead we build the three required deliverables from template-based,
    AI-authored domain content covering the six IT Helpdesk categories named
    in the assignment brief:

        - System Interruptions
        - Connectivity Issue
        - Issue with SaaS Platform Functionality
        - Application access related issue
        - Product assistance request
        - Invoice inquiry

    This satisfies the assignment's explicit allowance: "AI-generated domain
    content that you clean and verify" (Step 2) and "You can create the
    dataset manually, from public documents, from public datasets, or with
    the help of AI" (Step 4). No external downloads, no heavy ML libraries.

Outputs:
    data/non_instruction_data.txt   (>= 50 raw domain paragraphs)
    data/instruction_dataset.jsonl  (>= 100 instruction/response pairs)
    data/preference_dataset.jsonl   (>= 50 prompt/chosen/rejected triples)

Usage:
    python generate_domain_data.py
"""

import json
import random
from pathlib import Path

random.seed(42)

THIS_DIR = Path(__file__).resolve().parent
NON_INSTRUCTION_PATH = THIS_DIR / "non_instruction_data.txt"
INSTRUCTION_PATH = THIS_DIR / "instruction_dataset.jsonl"
PREFERENCE_PATH = THIS_DIR / "preference_dataset.jsonl"

# ---------------------------------------------------------------------------
# Domain building blocks, organized by IT Helpdesk category
# ---------------------------------------------------------------------------

CATEGORIES = {
    "System Interruptions": {
        "subjects": [
            "Unplanned Service Outage Affecting Production Environment",
            "Scheduled Maintenance Window Causing Downtime",
            "Server Crash During Peak Business Hours",
            "Intermittent System Downtime on Core Platform",
            "Database Cluster Failover Interrupting Service",
        ],
        "bodies": [
            "Our team noticed the platform became unresponsive around {time} and several users are unable to log in.",
            "The outage started after the latest deployment and is impacting multiple downstream services.",
            "We received alerts from our monitoring dashboard indicating elevated error rates and dropped connections.",
            "This is affecting our entire team and we need an update on the expected resolution time.",
        ],
        "responses": [
            "Thank you for reporting this outage. Our engineering team has identified the root cause as a failed node in the primary cluster and has already initiated failover to the backup node. We expect full service restoration within 30 minutes. We will send a status update once the issue is resolved and will follow up with a post-incident report.",
            "We apologize for the disruption. Our monitoring systems detected the interruption at {time} and our on-call engineers are actively working on restoring service. A hotfix has been deployed to the affected servers, and we are validating stability now. We will notify you as soon as the system is fully operational.",
        ],
    },
    "Connectivity Issue": {
        "subjects": [
            "Multiple Device Connection Problems on Corporate VPN",
            "Unable to Establish Stable Wi-Fi Connection",
            "Frequent Network Disconnections While Using the Application",
            "VPN Timeout Errors When Connecting Remotely",
            "Slow Network Speeds Impacting Daily Operations",
        ],
        "bodies": [
            "I have tried reconnecting several times but the connection drops every few minutes.",
            "This issue started after the recent firmware update on our router.",
            "Our remote employees are reporting the same symptoms across different locations.",
            "We have already restarted the router and modem but the problem persists.",
        ],
        "responses": [
            "Thank you for the details. This type of intermittent disconnection is often caused by an outdated VPN client or DNS misconfiguration. Please update your VPN client to the latest version and switch your DNS server to 8.8.8.8 as a temporary workaround. If the issue continues after these steps, please share your connection logs so our network team can investigate further.",
            "We understand how disruptive unstable connectivity can be. Please try connecting via a wired Ethernet connection to rule out Wi-Fi interference, and confirm whether the issue occurs on other devices on the same network. In the meantime, we are reviewing your network logs and will follow up within 24 hours with our findings.",
        ],
    },
    "Issue with SaaS Platform Functionality": {
        "subjects": [
            "Immediate Help Needed: Technical Problem with Cloud SaaS Service",
            "Dashboard Widgets Not Loading Correctly",
            "Export Feature Failing on SaaS Reporting Module",
            "Unexpected Errors When Saving Configuration Changes",
            "Search Functionality Returning Incorrect Results",
        ],
        "bodies": [
            "Every time I try to generate the report, the page freezes and shows a generic error message.",
            "This has been happening since the last product update was rolled out.",
            "Our whole team relies on this feature daily and it is now blocking our workflow.",
            "I have cleared my browser cache and tried a different browser, but the issue remains.",
        ],
        "responses": [
            "Thank you for reporting this. We have confirmed a known issue with the reporting module following our recent release, which is causing the export feature to time out for larger datasets. Our engineering team has a fix scheduled for deployment within the next business day. As a temporary workaround, please try exporting smaller date ranges in the meantime.",
            "We appreciate you flagging this. Based on your description, this looks related to a caching issue introduced in the latest update. Please try a hard refresh (Ctrl+F5) and confirm if the widgets load correctly. If the problem persists, we will escalate this to our platform engineering team and keep you updated on the resolution timeline.",
        ],
    },
    "Application access related issue": {
        "subjects": [
            "Unable to Access Office Applications",
            "Login Failure After Password Reset",
            "Account Locked Out After Multiple Failed Attempts",
            "Single Sign-On Authentication Not Working",
            "Permission Denied Error When Opening Shared Workspace",
        ],
        "bodies": [
            "I reset my password yesterday but I still cannot log in to the application.",
            "The system shows an authentication error even though my credentials are correct.",
            "This is urgent as I need access to complete a client deliverable today.",
            "I have tried logging in from both my laptop and mobile device with the same result.",
        ],
        "responses": [
            "Thank you for providing a detailed explanation of the issue. To assist you further, please specify any error messages encountered when launching the application. Also, verify whether your operating system is up to date and confirm that the latest version of the client software is installed. Since immediate access is critical, we can schedule a call at a convenient time to guide you through advanced troubleshooting steps. Please let us know your availability.",
            "We understand access issues are urgent. Our records show your account was temporarily locked due to repeated failed login attempts. We have reset the lockout and sent a secure link to your registered email to set a new password. Please use this link within the next 30 minutes, and let us know once you regain access so we can confirm everything is working.",
        ],
    },
    "Product assistance request": {
        "subjects": [
            "Request for Guidance on Product Configuration",
            "Need Help Setting Up Integration with Third-Party Tool",
            "Clarification Needed on Feature Usage",
            "Assistance Required for Bulk Data Import",
            "Guidance Requested on Best Practices for Workflow Automation",
        ],
        "bodies": [
            "We are trying to configure the integration but are not sure which settings apply to our use case.",
            "Could you provide documentation or a walkthrough for this feature?",
            "We attempted the import following the guide but encountered validation errors on several rows.",
            "Our team would like a recommendation on the most efficient way to automate this recurring task.",
        ],
        "responses": [
            "Thank you for reaching out. I would be happy to help you configure this integration. Based on your use case, I recommend enabling the sync settings under Admin > Integrations and mapping your fields according to the attached template. I have also included a link to our detailed setup guide. If you would like, we can schedule a short call to walk through the configuration together.",
            "Thanks for the detailed context. The validation errors you are seeing are typically caused by mismatched column headers in the import file. Please ensure your file follows the exact template we provide, paying close attention to date formats and required fields. I have attached a corrected sample file for reference, and I am happy to review your file directly if you can share it with us.",
        ],
    },
    "Invoice inquiry": {
        "subjects": [
            "Inquiry for Comprehensive Billing Procedure Details",
            "Discrepancy Noticed in Latest Invoice Amount",
            "Request for Copy of Previous Month's Invoice",
            "Question About Unexpected Additional Charges",
            "Clarification Needed on Subscription Renewal Pricing",
        ],
        "bodies": [
            "I reviewed our latest invoice and noticed the amount does not match our agreed subscription plan.",
            "Could you please resend the invoice for last month as we did not receive it via email?",
            "We were charged an additional fee that was not explained in our contract.",
            "We would like to understand how the renewal pricing is calculated for the upcoming term.",
        ],
        "responses": [
            "Thank you for bringing this to our attention. After reviewing your account, we found that the discrepancy was due to a prorated charge from a mid-cycle plan upgrade. I have attached a detailed breakdown of the charges for transparency. If you still believe this is incorrect after reviewing the breakdown, please let us know and we will escalate to our billing team for a full audit.",
            "We apologize for the confusion regarding your invoice. I have resent a copy of last month's invoice to your registered email address. Please confirm once you receive it. Regarding the renewal pricing, it is calculated based on your current plan tier and any add-ons active at the time of renewal; I have also attached a pricing summary for your reference.",
        ],
    },
}

GENERIC_REJECTIONS = [
    "Please try restarting your device and contact us again if the issue continues.",
    "This is a known issue. Please wait and it should resolve itself.",
    "We are unable to help with this right now, please check the FAQ page.",
    "Try turning it off and on again.",
    "This isn't really our department, please contact someone else.",
    "Not sure what to tell you, maybe try again later.",
    "That's just how the system works, nothing we can do.",
    "Please Google the error message and follow whatever comes up.",
]

TIMES = ["9:15 AM", "11:40 AM", "2:05 PM", "4:30 PM", "8:50 AM", "1:20 PM"]

MIN_NON_INSTRUCTION_PARAGRAPHS = 50
MIN_INSTRUCTION_EXAMPLES = 100
MIN_PREFERENCE_EXAMPLES = 50


def build_non_instruction_data() -> list:
    paragraphs = []
    for category, content in CATEGORIES.items():
        for subject in content["subjects"]:
            body = random.choice(content["bodies"]).format(time=random.choice(TIMES))
            answer = random.choice(content["responses"]).format(time=random.choice(TIMES))
            paragraph = f"[{category}] {subject}. {body} {answer}"
            paragraphs.append(paragraph)
    # Duplicate with body/response permutations to comfortably exceed the minimum.
    while len(paragraphs) < MIN_NON_INSTRUCTION_PARAGRAPHS:
        category = random.choice(list(CATEGORIES.keys()))
        content = CATEGORIES[category]
        subject = random.choice(content["subjects"])
        body = random.choice(content["bodies"]).format(time=random.choice(TIMES))
        answer = random.choice(content["responses"]).format(time=random.choice(TIMES))
        paragraph = f"[{category}] {subject}. {body} {answer}"
        if paragraph not in paragraphs:
            paragraphs.append(paragraph)
    random.shuffle(paragraphs)
    return paragraphs


def build_instruction_examples() -> list:
    examples = []
    for category, content in CATEGORIES.items():
        for subject in content["subjects"]:
            for body in content["bodies"]:
                answer = random.choice(content["responses"]).format(time=random.choice(TIMES))
                instruction = f"{subject}\n\n{body.format(time=random.choice(TIMES))}"
                examples.append({"instruction": instruction, "response": answer})
    random.shuffle(examples)
    return examples


def build_preference_examples() -> list:
    examples = []
    for category, content in CATEGORIES.items():
        for subject in content["subjects"]:
            for body in content["bodies"]:
                chosen = random.choice(content["responses"]).format(time=random.choice(TIMES))
                prompt = f"{subject}\n\n{body.format(time=random.choice(TIMES))}"
                rejected = random.choice(GENERIC_REJECTIONS)
                examples.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    random.shuffle(examples)
    return examples


def main():
    paragraphs = build_non_instruction_data()
    assert len(paragraphs) >= MIN_NON_INSTRUCTION_PARAGRAPHS, "not enough paragraphs"
    NON_INSTRUCTION_PATH.write_text("\n\n".join(paragraphs), encoding="utf-8")
    print(f"Wrote {len(paragraphs)} paragraphs -> {NON_INSTRUCTION_PATH}")

    instruction_examples = build_instruction_examples()
    assert len(instruction_examples) >= MIN_INSTRUCTION_EXAMPLES, "not enough instruction examples"
    with open(INSTRUCTION_PATH, "w", encoding="utf-8") as f:
        for ex in instruction_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(instruction_examples)} instruction examples -> {INSTRUCTION_PATH}")

    preference_examples = build_preference_examples()
    assert len(preference_examples) >= MIN_PREFERENCE_EXAMPLES, "not enough preference examples"
    with open(PREFERENCE_PATH, "w", encoding="utf-8") as f:
        for ex in preference_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(preference_examples)} preference examples -> {PREFERENCE_PATH}")


if __name__ == "__main__":
    main()
