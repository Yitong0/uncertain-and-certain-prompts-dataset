from pathlib import Path
import pandas as pd
from utils import article


def make_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


clear_factual_contradictions = [
    # geometry / math / logic
    "square circle",
    "one-sided polygon",
    "edgeless polygon",
    "scalene equilateral triangle",
    "finite unbounded region",
    "triangular sphere",
    "non three-sided triangle",
    "rectangular circle",
    "triangular circle",
    "two-dimensional volume",
    "one-dimensional surface",
    "obtuse right angle",
    "acute obtuse angle",
    "convergent divergent series",
    "continuous discrete function",
    "bijective non-invertible function",
    "square sphere",
    "circular cube",
    "rectangular sphere",
    "dimensionless cube",
    "parallel intersecting lines",
    "perpendicular parallel lines",
    "curved straight line",
    "zero-dimensional solid",
    "two-sided triangle",
    "uncountable finite set",
    "open closed interval",
    "acyclic cycle",
    "even odd integer",
    "rational irrational number",
    "negative natural number",
    "empty set with elements",
    "irrational integer",
    "disconnected connected graph",
    "singleton set of two elements",
    "smallest positive real number",
    

    # biological / perception
    "living corpse",
    "dead living being",
    "blind seer",
    "silent speaker",
    "speaking mute",
    "hearing deaf listener",
    "breathing corpse",
    "sterile reproduction",
    "asexual sexual reproduction",
    "anaerobic obligate aerobe",
    "non-dividing mitosis",
    "transparent opaque object",

    # physical / sensory
    "rigid fluid",
    "illuminated shadow",
    "weightless mass",
    "sub-zero absolute temperature",
    "absolute zero warmth",
    "transparent darkness",
    "solid smoke",
    "liquid stone",
    "dry liquid",
    "colorless visible spectrum",
    "toneless melody",
    "voiceless vocal",
    "silent phoneme",
    "unison interval",
    "legal crime",
    "free prison",
    "hot ice",
    "cold flame",
]


ambiguous_concept_contradictions = [
    # logic / linguistic / abstract
    "circled cube",
    "closed open surface",
    "non-self-intersecting loop",
    "false tautology",
    "valid fallacy",
    "non-self-intersecting figure-eight",
    "integer fraction",
    "real imaginary number",
    "finite infinity",
    "empty fullness",
    "greatest integer",
    "sound unsound argument",
    "improper subset excluding itself",
    "finite uncountable set",
    "non-empty null set",
    "noun-less noun phrase",
    "wordless sentence",
    "verbless predicate",
    "meaning-free semantic content",
    "grammarless grammatical sentence",
    "contentless proposition",
    "meaningless meaning",
    "voiceless whisper",
    "sleeping insomniac",
    "awake sleeper",
    "seeing blind man",
    "listening deaf person",
    "watching blind observer",
    "painless nociception",
    "sighted blind spot",
    "cold heat",
    "frictionless grip",
    "superconducting resistance",
    "heavy light weight",
    "instantaneous average",
    "odorless perfume scent",
    "invisible brightness",
    "soundless music",
    "standing run",
    "noiseless transmission through noise",
    "harmonic dissonance",

    # abstract/social/conceptual
    "ordered entropy",
    "conscious unconsciousness",
    "consensual dictatorship",
    "oppressive freedom",
    "single married person",
    "widowed living husband",
    "certain uncertainty",
    "unchanging change",
    "voluntary obligation",
    "victimless crime with a victim",
    "stateless citizen",
    "married bachelor",
    "colorblind color distinguisher",
    "omnipotent limited being",
    "omniscient ignorant mind",
    "halted infinite loop",
    "voluntary forced consent",
    "identical distinct entities",
    "humble arrogance",
    "peaceful warfare",
    "gentle violence",
    "honest corruption",
    "sacred profanity",
    "identical non-identical twins",
    "local global minimum",
    "structured pure randomness",
    "painless agony",
    "blind precise observer",
    "effortless laborious stillness",
    "silent acoustic resonance",
    "stationary moving reference",
    "involuntary deliberate reflex",
    "structured random noise",
    "mute eloquent speaker",
    "hollow substantial argument",
    "absent minded awarenes",
    "predictable pure chaos",
    "odorless distinct aroma",
    "rigid perfectly flowing liquid",
    "dark vivid illumination",
    "faithful betrayal",
    "gentle extreme violence",
    "humble complete arrogance",
    "brave cowardice",
    "joyful profound grief",
    "reckless careful action",
    "old fresh novelty",
    "shallow deep wisdom",
    "joyful desolation",
    "safe danger",
    "gentle wrestling",
    "sharp softness",
    "living fossil",
    "merciful tyrant",
    "mortal immortality",
    "finite eternity",
    "warm indifference",
    "dark radiance",
    "heavy grace",
    "gentle fierce storm",
    "rough fine silk",
    "deep flat surface",
    "smooth sharp edge",
    "gentle fierce roar",
]


def generate_logical_contradictions():
    rows = []
    idx = 1

    contradiction_groups = {
        "clear_factual_contradiction": clear_factual_contradictions,
        "ambiguous_concept_contradiction": ambiguous_concept_contradictions,
    }

    for subcategory_name, contradiction_list in contradiction_groups.items():
        for concept in contradiction_list:
            prompt = f"{article(concept)} {concept}"

            rows.append(
                {
                    "prompt_id": f"unc_adv_{idx:05d}",
                    "prompt_text": prompt,
                    "template_type": "a_plus_concept",
                    "category": "uncertain",
                    "subcategory": subcategory_name,
                    "base_prompt_id": None,
                    "source_prompt_text": concept,
                    "generation_method": "logical_contradiction_template",
                    "contradiction": concept,
                }
            )

            idx += 1

    df = pd.DataFrame(rows)

    # Remove exact duplicate prompt texts if any remain
    df = df.drop_duplicates(subset=["prompt_text"]).reset_index(drop=True)

    return df


def main():
    output_dir = Path("data/generated")
    output_path = output_dir / "adversarial_nonsense_variants.csv"

    make_output_dir(output_dir)

    df = generate_logical_contradictions()

    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} adversarial nonsense prompts")
    print("\nCounts by subcategory:")
    print(df["subcategory"].value_counts())
    print("\nSample rows:")
    print(df.head(20))


if __name__ == "__main__":
    main()