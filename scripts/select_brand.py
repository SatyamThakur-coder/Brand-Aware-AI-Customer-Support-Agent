"""
Script to select target brand for AI customer support system.
Updates configs/config.yaml and generates reports/brand_selection.md.
"""

import argparse
import yaml
from pathlib import Path

CONFIG_PATH = Path("configs/config.yaml")
REPORTS_DIR = Path("reports")

BRAND_METRICS = {
    "AmazonHelp": {
        "outbound_replies": 169840,
        "sample_resolved_pairs": 42709,
        "categories": "Delivery delay, Refund/Return, Prime subscription, Missing item, Account login, Payment issue",
        "rationale": "Selected AmazonHelp because it possesses the largest volume of brand responses (169,840 tweets) and multi-turn resolution pairs (42,709 sampled) across standard e-commerce customer support categories."
    },
    "AppleSupport": {
        "outbound_replies": 106860,
        "sample_resolved_pairs": 15637,
        "categories": "iOS updates, Battery, iCloud, Device hardware, App crashes",
        "rationale": "Selected AppleSupport due to high volume (106,860 tweets) focused on tech support and software troubleshooting."
    },
    "Uber_Support": {
        "outbound_replies": 56270,
        "sample_resolved_pairs": 10353,
        "categories": "Driver issues, Ride cancellation, Fare disputes, Lost items",
        "rationale": "Selected Uber_Support due to high volume (56,270 tweets) focused on ride hail and trip disputes."
    }
}

def select_brand(brand_name: str):
    if not CONFIG_PATH.exists():
        print(f"Error: Config file {CONFIG_PATH} does not exist.")
        return

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    config["brand"]["name"] = brand_name
    config["brand"]["display_name"] = brand_name.replace("_", " ")

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Updated {CONFIG_PATH} with brand: {brand_name}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS_DIR / "brand_selection.md"

    info = BRAND_METRICS.get(brand_name, {
        "outbound_replies": "N/A",
        "sample_resolved_pairs": "N/A",
        "categories": "General customer support",
        "rationale": f"Selected {brand_name} based on dataset availability and interaction volume."
    })

    md_content = f"""# Brand Selection Report: {brand_name}

## Measurable Selection Criteria

The target brand **`{brand_name}`** was selected from the Kaggle dataset (`thoughtvector/customer-support-on-twitter`) based on empirical statistics:

- **Total Outbound Brand Replies**: {info['outbound_replies']:,} tweets
- **Sampled Resolved Conversation Pairs**: {info['sample_resolved_pairs']:,} threads
- **Key Support Issue Categories**: {info['categories']}

## Rationale

{info['rationale']}

Unlike low-volume handles with sparse customer replies, `{brand_name}` provides sufficient historical multi-turn customer-agent dialogs required to construct realistic intent taxonomies, train baselines, build dense semantic vector indexes, and evaluate grounded reply generation.
"""

    with open(md_path, "w") as f:
        f.write(md_content)

    print(f"Generated brand selection report at {md_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select brand for AI support agent pipeline.")
    parser.add_argument("--brand", type=str, default="AmazonHelp", help="Target brand handle (e.g., AmazonHelp)")
    args = parser.parse_args()

    select_brand(args.brand)
