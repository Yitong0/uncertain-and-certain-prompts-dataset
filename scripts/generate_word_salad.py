from pathlib import Path
import pandas as pd
import random

from utils import make_word_salad


def make_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def generate_word_salad_variants(df_base: pd.DataFrame, seed=42) -> pd.DataFrame:
    random.seed(seed)

    rows = []
    variant_id = 1

    for _, row in df_base.iterrows():
        base_prompt = row["prompt_text"]
        base_id = row["prompt_id"]

        new_prompt, original_words = make_word_salad(base_prompt)

        rows.append(
            {
                "prompt_id": f"unc_wordsalad_{variant_id:05d}",
                "prompt_text": new_prompt,
                "template_type": row["template_type"],
                "category": "uncertain",
                "subcategory": "word_salad",
                "base_prompt_id": base_id,
                "source_prompt_text": base_prompt,
                "generation_method": "word_shuffle",
                "original_word_order": " ".join(original_words),
            }
        )

        variant_id += 1

    return pd.DataFrame(rows)


def main():
    base_path = Path("data/raw/base_prompts.csv")
    output_dir = Path("data/generated")
    output_path = output_dir / "word_salad_variants.csv"

    make_output_dir(output_dir)

    df_base = pd.read_csv(base_path)

    df_word_salad = generate_word_salad_variants(df_base)

    df_word_salad.to_csv(output_path, index=False)

    print(f"Saved {len(df_word_salad)} word salad variants to {output_path}")


if __name__ == "__main__":
    main()