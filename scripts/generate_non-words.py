from pathlib import Path
import pandas as pd
import random

from utils import replace_with_pseudoword


def make_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def generate_nonword_variants(df_base: pd.DataFrame, seed=42):

    random.seed(seed)

    rows = []
    variant_id = 1

    for _, row in df_base.iterrows():

        base_prompt = row["prompt_text"]
        base_id = row["prompt_id"]

        new_prompt, original_word, pseudoword = replace_with_pseudoword(base_prompt)

        rows.append(
            {
                "prompt_id": f"unc_nonword_{variant_id:05d}",
                "prompt_text": new_prompt,
                "template_type": row["template_type"],
                "category": "uncertain",
                "subcategory": "non_word",
                "base_prompt_id": base_id,
                "source_prompt_text": base_prompt,
                "generation_method": "pseudoword_replacement",
                "original_word": original_word,
                "pseudoword": pseudoword,
            }
        )

        variant_id += 1

    return pd.DataFrame(rows)


def main():

    base_path = Path("data/raw/base_prompts.csv")
    output_dir = Path("data/generated")
    output_path = output_dir / "nonword_variants.csv"

    make_output_dir(output_dir)

    df_base = pd.read_csv(base_path)

    df_nonword = generate_nonword_variants(df_base)

    df_nonword.to_csv(output_path, index=False)

    print(f"Saved {len(df_nonword)} non-word variants to {output_path}")


if __name__ == "__main__":
    main()