"""
Purpose: Generate synthetic test privacy policies representing varying degrees of regulatory compliance to evaluate the PrERT-CNM engine.
Goal: Produce highly deterministic testing data (Perfect, Mixed, and Absolute Violation) to explicitly test the mathematical boundaries of the Bayesian Variable Elimination and DeBERTa True/False Positives.
Scalability & Innovation: Relying on generic public datasets often introduces noise and unmapped edge cases. By deterministically generating edge-case policies, we control the exact latent features the model should be extracting, allowing us to enforce strict Unit Testing against our ISO/GDPR mappings.
"""
import os
from pathlib import Path


def _build_long_policy(title: str, paragraphs: list[str], repetitions: int = 3) -> str:
    sections = [f"Policy {title}:\n"]
    for cycle in range(1, repetitions + 1):
        sections.append(f"Section {cycle}: Operational Clauses")
        for paragraph in paragraphs:
            sections.append(paragraph)
        sections.append("")
    return "\n\n".join(sections).strip()


def _build_multi_policy_document(policies: list[str]) -> str:
    divider = "\n\n" + ("-" * 48) + "\n\n"
    return divider.join(policies)

def generate_policies():
    base_dir = Path(__file__).parent.parent / "data" / "test_policies"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    perfect = (
        "We collect only the minimum necessary data to provide our core services. "
        "User data is never sold to third parties under any circumstances. "
        "All data at rest and in transit is secured using AES-256 encryption. "
        "Upon request or account deletion, all associated personal data is permanently wiped within 30 days. "
        "We obtain explicit, opt-in consent before collecting geolocation or biometric data."
    )
    
    mixed = (
        "We collect your personal information to optimize your experience. "
        "While we secure your password using standard encryption, we may share anonymized browsing habits with our marketing partners. "
        "You can request account deletion, but we retain certain data for internal analytics indefinitely. "
        "Geolocation is collected while the app is in use."
    )
    
    violation = (
        "By using this service, you grant us irrevocable rights to collect, store, and sell your personal data, including geolocation and browsing history, to our third-party advertising network. "
        "We do not employ encryption on our internal databases to ensure fast querying speeds. "
        "We retain all user data indefinitely, even if you delete your account. "
        "We automatically opt you in to all data collection streams upon signup without requiring explicit consent."
    )
    
    with open(base_dir / "perfect_compliance.txt", "w", encoding="utf-8") as f:
        f.write(perfect)
        
    with open(base_dir / "mixed_compliance.txt", "w", encoding="utf-8") as f:
        f.write(mixed)
        
    with open(base_dir / "absolute_violation.txt", "w", encoding="utf-8") as f:
        f.write(violation)

    perfect_long_paragraphs = [
        "We collect only the minimum information required to authenticate and deliver requested services, and we avoid processing unrelated behavioral attributes.",
        "Every data transfer channel uses modern encryption in transit, and internal records are encrypted at rest using role-restricted key management controls.",
        "Users can request correction or deletion of personal information, and all linked profile artifacts are purged according to a fixed deletion schedule.",
        "No personal information is sold, licensed, or exchanged with third parties for advertising monetization.",
        "High-risk attributes such as geolocation and biometrics remain opt-in and are disabled by default for every account."
    ]

    mixed_long_paragraphs = [
        "We collect account and activity information to personalize experiences and maintain product quality, but retention controls differ across internal teams.",
        "Credential data is protected with standard encryption practices, while selected telemetry signals are retained for model diagnostics over extended windows.",
        "Users can request deletion workflows, but some analytical records are kept for ongoing experimentation and trend analysis.",
        "We may share aggregated usage insights with strategic marketing partners, while claiming those datasets are anonymized.",
        "Location-based services are active during feature use and can influence recommendations even when users are not explicitly reviewing privacy toggles."
    ]

    violation_long_paragraphs = [
        "By continuing to use this platform, you grant irrevocable rights for us to collect, store, and commercialize your personal data with advertising networks.",
        "Internal databases do not use encryption controls because plaintext processing is considered faster for business analytics and operational reporting.",
        "All historical account records are retained indefinitely, including post-deletion traces, to support indefinite profiling and retrospective modeling.",
        "New users are automatically opted into every data stream, including geolocation, browsing, and derived behavioral scoring, without explicit consent.",
        "Data sharing permissions are broad by default and may include third-party enrichment vendors with limited transparency disclosures."
    ]

    perfect_long = _build_long_policy("1 - Perfect Compliance", perfect_long_paragraphs, repetitions=4)
    mixed_long = _build_long_policy("2 - Mixed Compliance", mixed_long_paragraphs, repetitions=4)
    violation_long = _build_long_policy("3 - Absolute Violation", violation_long_paragraphs, repetitions=4)

    multi_policy_long = _build_multi_policy_document([perfect_long, mixed_long, violation_long])

    with open(base_dir / "perfect_longer.txt", "w", encoding="utf-8") as f:
        f.write(perfect_long)

    with open(base_dir / "mixed_longer.txt", "w", encoding="utf-8") as f:
        f.write(mixed_long)

    with open(base_dir / "absolute_violation_longer.txt", "w", encoding="utf-8") as f:
        f.write(violation_long)

    with open(base_dir / "multi_policy_long.txt", "w", encoding="utf-8") as f:
        f.write(multi_policy_long)
        
    print(f"Successfully generated synthetic policies in {base_dir}")

if __name__ == "__main__":
    generate_policies()
