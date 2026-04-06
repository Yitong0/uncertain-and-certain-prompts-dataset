from pathlib import Path
import pandas as pd
import random

from utils import article


def make_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def generate_adversarial_nonsense(df_base: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)

    nonsense_attributes = [
        "liquid",
        "edible",
        "sleeping",
        "singing",
        "melting",
        "transparent",
    ]

    rows = []
    variant_id = 1

    for _, row in df_base.iterrows():
        prompt = row["prompt_text"]
        template_type = row["template_type"]
        base_id = row["prompt_id"]

        if template_type != "attribute_object":
            continue

        words = prompt.split()

        if len(words) < 3:
            continue

        obj = words[2]
        new_attr = random.choice(nonsense_attributes)
        new_prompt = f"{article(new_attr)} {new_attr} {obj}"

        rows.append(
            {
                "prompt_id": f"unc_adv_{variant_id:05d}",
                "prompt_text": new_prompt,
                "template_type": template_type,
                "category": "uncertain",
                "subcategory": "adversarial_nonsense",
                "base_prompt_id": base_id,
                "source_prompt_text": prompt,
                "generation_method": "replace_attribute_with_nonsense_attribute",
                "original_object": obj,
                "nonsense_attribute": new_attr,
            }
        )

        variant_id += 1

    return pd.DataFrame(rows)


def main():
    base_path = Path("data/raw/base_prompts.csv")
    output_dir = Path("data/generated")
    output_path = output_dir / "adversarial_nonsense_variants.csv"

    make_output_dir(output_dir)

    df_base = pd.read_csv(base_path)

    df_adv = generate_adversarial_nonsense(df_base)

    df_adv.to_csv(output_path, index=False)

    print(f"Saved {len(df_adv)} adversarial nonsense variants to {output_path}")
    print(df_adv.head(20))


if __name__ == "__main__":
    main()