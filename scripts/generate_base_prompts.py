from pathlib import Path
import itertools
import pandas as pd
from utils import article


def make_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_base_prompts() -> list[dict]:
    objects = [
        "dog", "cat", "chair", "table", "cup", "bottle",
        "bicycle", "car", "rack", "apple", "banana",
        "book", "phone", "laptop", "camera", "backpack", "television"
    ]

    colors = [
        "red", "blue", "green", "yellow", "black", "white",
        "purple", "orange", "pink", "brown", "gray", "magenta", "lime", "cyan"
    ]

    attributes = [
        "small", "big", "old", "new", "wooden", "plastic", "glass", "stone",
        "metal", "shiny", "dirty", "clean", "wet", "dry", "heavy", "light", "round"
    ]

    places = [
        "room", "street", "park", "kitchen", "beach", "office", "garden", "forest", "desert", 
        "city", "village", "restaurant", "cafe", "library", "school", "market", "farm", "field",
        "airport", "station"

    ]

    surfaces = [
        "table", "floor", "desk", "shelf", "bed"
    ]

    rows = []
    prompt_id = 1

    # "a photo of a [object]"
    for obj in objects:
        rows.append(
            {
                "prompt_id": f"base_{prompt_id:04d}",
                "prompt_text": f"a photo of {article(obj)} {obj}",
                "template_type": "photo_of_object",
                "category": "certain",
                "base_prompt_id": "",
                "generation_method": "template: a photo of a [object]",
            }
        )
        prompt_id += 1

    # "a [color] [object]""
    for color, obj in itertools.product(colors, objects):
        rows.append(
            {
                "prompt_id": f"base_{prompt_id:04d}",
                "prompt_text": f"{article(color)} {color} {obj}",
                "template_type": "color_object",
                "category": "certain",
                "base_prompt_id": "",
                "generation_method": "template: a [color] [object]",
            }
        )
        prompt_id += 1

    # "a [attribute] [object]"
    for attr, obj in itertools.product(attributes, objects):
        rows.append(
            {
                "prompt_id": f"base_{prompt_id:04d}",
                "prompt_text": f"{article(attr)} {attr} {obj}",
                "template_type": "attribute_object",
                "category": "certain",
                "base_prompt_id": "",
                "generation_method": "template: a [attribute] [object]",
            }
        )
        prompt_id += 1

    # "a [object] on a [surface]"
    for obj, surface in itertools.product(objects, surfaces):
        rows.append(
            {
                "prompt_id": f"base_{prompt_id:04d}",
                "prompt_text": f"{article(obj)} {obj} on {article(surface)} {surface}",
                "template_type": "object_on_surface",
                "category": "certain",
                "base_prompt_id": "",
                "generation_method": "template: a [object] on a [surface]",
            }
        )
        prompt_id += 1

    # "two [objects] in a [place]"
    for obj, place in itertools.product(objects, places):
        # simple pluralization by adding "s"
        rows.append(
            {
                "prompt_id": f"base_{prompt_id:04d}",
                "prompt_text": f"two {obj}s in {article(place)} {place}",
                "template_type": "two_objects_in_place",
                "category": "certain",
                "base_prompt_id": "",
                "generation_method": "template: two [objects] in a [place]",
            }
        )
        prompt_id += 1

    return rows


def main() -> None:
    output_dir = Path("data/raw")
    make_output_dir(output_dir)

    rows = build_base_prompts()
    df = pd.DataFrame(rows)

    # remove duplicates just in case
    df = df.drop_duplicates(subset=["prompt_text"]).reset_index(drop=True)

    output_path = output_dir / "base_prompts.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} base prompts to: {output_path}")


if __name__ == "__main__":
    main()