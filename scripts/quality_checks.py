from pathlib import Path
import pandas as pd


DATA_DIR = Path("data")


def load_all_datasets():
    files = []

    # certain prompts
    files.append(DATA_DIR / "certain" / "base_prompts.csv")

    # uncertain prompts
    uncertain_dir = DATA_DIR / "uncertain"

    files += list(uncertain_dir.glob("*.csv"))

    dfs = []

    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    return full_df


def check_duplicates(df):

    duplicates = df[df.duplicated(subset=["prompt_text"], keep=False)]

    print("\nTOTAL PROMPTS:", len(df))
    print("UNIQUE PROMPTS:", df["prompt_text"].nunique())

    if duplicates.empty:
        print("\nNo duplicate prompt_text found across datasets")
    else:
        print("\nDUPLICATES FOUND!")

        print("\nNumber of duplicated prompts:", duplicates["prompt_text"].nunique())

        print("\nExamples:")
        print(duplicates[["prompt_text", "source_file"]].sort_values("prompt_text").head(20))

        duplicates.to_csv("outputs/duplicate_prompts.csv", index=False)

        print("\nFull duplicate list saved to outputs/duplicate_prompts.csv")


def main():

    df = load_all_datasets()

    check_duplicates(df)


if __name__ == "__main__":
    main()