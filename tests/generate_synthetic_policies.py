"""
Purpose: Generate synthetic test privacy policies representing varying degrees of regulatory compliance to evaluate the PrERT-CNM engine.
Goal: Produce highly deterministic testing data (Perfect, Mixed, and Absolute Violation) to explicitly test the mathematical boundaries of the Bayesian Variable Elimination and DeBERTa True/False Positives.
Scalability & Innovation: Relying on generic public datasets often introduces noise and unmapped edge cases. By deterministically generating edge-case policies, we control the exact latent features the model should be extracting, allowing us to enforce strict Unit Testing against our ISO/GDPR mappings.
"""
import os
from pathlib import Path

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
        
    print(f"Successfully generated synthetic policies in {base_dir}")

if __name__ == "__main__":
    generate_policies()
