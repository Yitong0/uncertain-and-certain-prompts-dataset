from pathlib import Path
import pandas as pd
import random

from utils import generate_gibberish_token


def make_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def generate_gibberish_variants(df_base: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)

    rows = []
    variant_id = 1

    for _, row in df_base.iterrows():
        base_prompt = row["prompt_text"]
        base_id = row["prompt_id"]
        template_type = row["template_type"]

        words = base_prompt.split()
        n_tokens = len(words)

        gibberish_tokens = [
            generate_gibberish_token() for _ in range(n_tokens)
        ]

        new_prompt = " ".join(gibberish_tokens)

        rows.append(
            {
                "prompt_id": f"unc_gib_{variant_id:05d}",
                "prompt_text": new_prompt,
                "template_type": template_type,
                "category": "uncertain",
                "subcategory": "gibberish",
                "base_prompt_id": base_id,
                "source_prompt_text": base_prompt,
                "generation_method": "random_gibberish_tokens",
                "token_count": n_tokens,
            }
        )

        variant_id += 1

    return pd.DataFrame(rows)


def main():
    base_path = Path("data/raw/base_prompts.csv")
    output_dir = Path("data/generated")
    output_path = output_dir / "gibberish_variants.csv"

    make_output_dir(output_dir)

    df_base = pd.read_csv(base_path)

    df_gib = generate_gibberish_variants(df_base)

    df_gib.to_csv(output_path, index=False)

    print(f"Saved {len(df_gib)} gibberish variants to {output_path}")
    print(df_gib.head(20))


if __name__ == "__main__":
    main()