from pathlib import Path
import pandas as pd


def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        print(f"Loaded: {path}")
        return pd.read_csv(path)
    else:
        print(f"Missing: {path}")
        return None


def print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def basic_checks(df: pd.DataFrame):
    print_header("BASIC DATASET CHECKS")

    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nColumns:")
    print(list(df.columns))

    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\nDuplicate prompt_text count:")
    print(df["prompt_text"].duplicated().sum())

    print("\nPrompt length stats (number of words):")
    prompt_lengths = df["prompt_text"].astype(str).str.split().str.len()
    print(prompt_lengths.describe())


def category_counts(df: pd.DataFrame):
    print_header("CATEGORY COUNTS")

    if "subcategory" in df.columns:
        print(df["subcategory"].value_counts(dropna=False))
    elif "category" in df.columns:
        print(df["category"].value_counts(dropna=False))
    else:
        print("No category/subcategory column found.")


def check_duplicates(df: pd.DataFrame):
    print_header("DUPLICATE PROMPTS")

    dupes = df[df["prompt_text"].duplicated(keep=False)].sort_values("prompt_text")
    print(f"Duplicate rows found: {len(dupes)}")

    if len(dupes) > 0:
        print("\nSample duplicate prompts:")
        print(dupes[["prompt_text"]].drop_duplicates().head(20))


def check_corrupted(df: pd.DataFrame):
    print_header("CORRUPTED PROMPTS CHECK")

    if "subcategory" not in df.columns:
        print("No subcategory column found.")
        return

    df_corrupt = df[df["subcategory"] == "corrupted_noisy"].copy()

    if df_corrupt.empty:
        print("No corrupted prompts found.")
        return

    same_as_source = (
        df_corrupt["prompt_text"].astype(str) == df_corrupt["source_prompt_text"].astype(str)
    ).sum()

    print(f"Corrupted rows: {len(df_corrupt)}")
    print(f"Corrupted prompts identical to source: {same_as_source}")

    print("\nSample corrupted prompts:")
    print(
        df_corrupt[
            ["source_prompt_text", "prompt_text", "generation_method"]
        ].head(10)
    )


def check_nonword(df: pd.DataFrame):
    print_header("NON-WORD PROMPTS CHECK")

    if "subcategory" not in df.columns:
        print("No subcategory column found.")
        return

    df_non = df[df["subcategory"] == "non_word"].copy()

    if df_non.empty:
        print("No non-word prompts found.")
        return

    print(f"Non-word rows: {len(df_non)}")

    if {"original_word", "pseudoword"}.issubset(df_non.columns):
        identical = (
            df_non["original_word"].astype(str).str.lower()
            == df_non["pseudoword"].astype(str).str.lower()
        ).sum()
        print(f"original_word == pseudoword count: {identical}")

    print("\nSample non-word prompts:")
    cols = [c for c in ["source_prompt_text", "prompt_text", "original_word", "pseudoword"] if c in df_non.columns]
    print(df_non[cols].head(10))


def check_word_salad(df: pd.DataFrame):
    print_header("WORD SALAD CHECK")

    if "subcategory" not in df.columns:
        print("No subcategory column found.")
        return

    df_ws = df[df["subcategory"] == "word_salad"].copy()

    if df_ws.empty:
        print("No word-salad prompts found.")
        return

    print(f"Word salad rows: {len(df_ws)}")

    if "source_prompt_text" in df_ws.columns:
        unchanged = (
            df_ws["prompt_text"].astype(str) == df_ws["source_prompt_text"].astype(str)
        ).sum()
        print(f"Word salad prompts identical to source: {unchanged}")

    print("\nSample word salad prompts:")
    cols = [c for c in ["source_prompt_text", "prompt_text"] if c in df_ws.columns]
    print(df_ws[cols].head(10))


def check_adversarial(df: pd.DataFrame):
    print_header("ADVERSARIAL NONSENSE CHECK")

    if "subcategory" not in df.columns:
        print("No subcategory column found.")
        return

    df_adv = df[df["subcategory"] == "adversarial_nonsense"].copy()

    if df_adv.empty:
        print("No adversarial-nonsense prompts found.")
        return

    print(f"Adversarial nonsense rows: {len(df_adv)}")

    print("\nSample adversarial nonsense prompts:")
    cols = [c for c in ["source_prompt_text", "prompt_text", "generation_method"] if c in df_adv.columns]
    print(df_adv[cols].head(10))


def check_gibberish(df: pd.DataFrame):
    print_header("GIBBERISH CHECK")

    if "subcategory" not in df.columns:
        print("No subcategory column found.")
        return

    df_gib = df[df["subcategory"] == "gibberish"].copy()

    if df_gib.empty:
        print("No gibberish prompts found.")
        return

    print(f"Gibberish rows: {len(df_gib)}")

    print("\nSample gibberish prompts:")
    cols = [c for c in ["source_prompt_text", "prompt_text", "token_count"] if c in df_gib.columns]
    print(df_gib[cols].head(10))


def save_summary(df: pd.DataFrame, output_path: Path):
    summary_rows = []

    summary_rows.append(
        {"metric": "total_rows", "value": len(df)}
    )
    summary_rows.append(
        {"metric": "duplicate_prompt_text_count", "value": df["prompt_text"].duplicated().sum()}
    )

    if "subcategory" in df.columns:
        counts = df["subcategory"].value_counts(dropna=False)
        for name, count in counts.items():
            summary_rows.append(
                {"metric": f"subcategory_count::{name}", "value": count}
            )

    prompt_lengths = df["prompt_text"].astype(str).str.split().str.len()
    summary_rows.extend(
        [
            {"metric": "prompt_length_mean", "value": prompt_lengths.mean()},
            {"metric": "prompt_length_min", "value": prompt_lengths.min()},
            {"metric": "prompt_length_max", "value": prompt_lengths.max()},
        ]
    )

    summary_df = pd.DataFrame(summary_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    print(f"\nSaved summary report to: {output_path}")


def main():
    data_dir = Path("data/generated")
    raw_dir = Path("data/raw")
    output_report = Path("outputs/statistics.csv")

    datasets = []

    # base prompts
    base_df = load_csv_if_exists(raw_dir / "base_prompts.csv")
    if base_df is not None:
        if "subcategory" not in base_df.columns:
            base_df["subcategory"] = "certain"
        datasets.append(base_df)

    # generated categories
    files = [
        "corrupted_variants.csv",
        "nonword_variants.csv",
        "word_salad_variants.csv",
        "adversarial_nonsense_variants.csv",
        "gibberish_variants.csv",
    ]

    for file_name in files:
        df_part = load_csv_if_exists(data_dir / file_name)
        if df_part is not None:
            datasets.append(df_part)

    if not datasets:
        print("No datasets found.")
        return

    df = pd.concat(datasets, ignore_index=True)

    basic_checks(df)
    category_counts(df)
    check_duplicates(df)
    check_corrupted(df)
    check_nonword(df)
    check_word_salad(df)
    check_adversarial(df)
    check_gibberish(df)

    save_summary(df, output_report)


if __name__ == "__main__":
    main()