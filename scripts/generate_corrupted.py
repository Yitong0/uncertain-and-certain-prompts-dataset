from pathlib import Path
import random
import pandas as pd

from utils import corrupt_prompt


def make_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_corrupted_variants(
    df_base: pd.DataFrame,
    methods: list[str],
    seed: int = 42
) -> pd.DataFrame:
    random.seed(seed)

    rows = []
    variant_id = 1

    for _, row in df_base.iterrows():
        base_prompt_id = row["prompt_id"]
        base_prompt_text = row["prompt_text"]

        for method in methods:
            corrupted_prompt, original_word = corrupt_prompt(base_prompt_text, method)

            rows.append(
                {
                    "prompt_id": f"unc_{variant_id:05d}",
                    "prompt_text": corrupted_prompt,
                    "template_type": row["template_type"],
                    "category": "uncertain",
                    "subcategory": "corrupted_noisy",
                    "base_prompt_id": base_prompt_id,
                    "source_prompt_text": base_prompt_text,
                    "generation_method": method,
                    "corrupted_word": original_word,
                }
            )
            variant_id += 1

    return pd.DataFrame(rows)


def main() -> None:
    input_path = Path("data/raw/base_prompts.csv")
    output_dir = Path("data/generated")
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

    # remove duplicates if two corruptions accidentally produce the same result
    df_corrupted = df_corrupted.drop_duplicates(
        subset=["prompt_text", "base_prompt_id", "generation_method"]
    ).reset_index(drop=True)

    df_corrupted.to_csv(output_path, index=False)

    print(f"Saved {len(df_corrupted)} corrupted variants to: {output_path}")


if __name__ == "__main__":
    main()