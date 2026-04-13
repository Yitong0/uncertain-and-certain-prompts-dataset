from pathlib import Path
import random
import pandas as pd

from utils import corrupt_prompt


def make_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def generate_corrupted_variants(df_base: pd.DataFrame, methods: list[str], seed: int = 42):
    random.seed(seed)

    rows = []
    variant_id = 1

    base_prompt_set = set(df_base["prompt_text"])
    used_corrupted_prompts = set()

    for _, row in df_base.iterrows():
        base_prompt = row["prompt_text"]
        base_id = row["prompt_id"]

        for method in methods:
            corrupted_prompt, original_word = corrupt_prompt(base_prompt, method)

            # reject unchanged
            if corrupted_prompt == base_prompt:
                continue

            # reject collisions with certain dataset
            if corrupted_prompt in base_prompt_set:
                continue

            # reject duplicates within corrupted dataset
            if corrupted_prompt in used_corrupted_prompts:
                continue

            used_corrupted_prompts.add(corrupted_prompt)

            rows.append(
                {
                    "prompt_id": f"unc_{variant_id:05d}",
                    "prompt_text": corrupted_prompt,
                    "template_type": row["template_type"],
                    "category": "uncertain",
                    "subcategory": "corrupted_noisy",
                    "base_prompt_id": base_id,
                    "source_prompt_text": base_prompt,
                    "generation_method": method,
                    "corrupted_word": original_word,
                }
            )

            variant_id += 1

    return pd.DataFrame(rows)


def main():
    input_path = Path("data/certain/base_prompts.csv")
    output_dir = Path("data/uncertain")
    output_path = output_dir / "corrupted_variants.csv"

    make_output_dir(output_dir)

    df_base = pd.read_csv(input_path)

    corruption_methods = [
        "delete_char",
        "insert_char",
        "substitute_char",
        "swap_adjacent_chars",
        "truncate_word",
    ]

    df_corrupted = generate_corrupted_variants(
        df_base=df_base,
        methods=corruption_methods,
        seed=42,
    )

    df_corrupted.to_csv(output_path, index=False)

    print(f"Saved {len(df_corrupted)} corrupted variants to: {output_path}")


if __name__ == "__main__":
    main()