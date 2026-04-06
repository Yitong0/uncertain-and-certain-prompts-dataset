import random
import string

def article(word: str) -> str:
    word = word.strip().lower()
    if word[0] in "aeiou":
        return "an"
    return "a"

ALPHABET = string.ascii_lowercase


def choose_word_index(words: list[str]) -> int | None:
    """
    Choose an index of a word that is safe to corrupt.
    Avoid very short function words like 'a', 'an', 'of', 'on', 'in', 'the', 'two'.
    """
    banned = {"a", "an", "of", "on", "in", "the", "two"}
    candidates = [
        i for i, w in enumerate(words)
        if w.lower() not in banned and len(w) >= 3
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def delete_char(word: str) -> str:
    """Delete one random character from a word."""
    if len(word) <= 1:
        return word
    idx = random.randrange(len(word))
    return word[:idx] + word[idx + 1:]


def insert_char(word: str) -> str:
    """Insert one random lowercase character into a word."""
    idx = random.randrange(len(word) + 1)
    char = random.choice(ALPHABET)
    return word[:idx] + char + word[idx:]


def substitute_char(word: str) -> str:
    """Replace one random character in a word with another lowercase character."""
    if len(word) == 0:
        return word
    idx = random.randrange(len(word))
    original = word[idx].lower()
    choices = [c for c in ALPHABET if c != original]
    new_char = random.choice(choices)
    return word[:idx] + new_char + word[idx + 1:]


def swap_adjacent_chars(word: str) -> str:
    """Swap two adjacent characters in a word."""
    if len(word) < 2:
        return word
    idx = random.randrange(len(word) - 1)
    chars = list(word)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


def truncate_word(word: str) -> str:
    """
    Remove the end of a word.
    Keeps at least 2 characters when possible.
    """
    if len(word) <= 3:
        return word[:2] if len(word) >= 2 else word
    cut = random.randrange(1, min(4, len(word) - 1))
    return word[:-cut]


def corrupt_word(word: str, method: str) -> str:
    """Apply one named corruption method to a word."""
    if method == "delete_char":
        return delete_char(word)
    if method == "insert_char":
        return insert_char(word)
    if method == "substitute_char":
        return substitute_char(word)
    if method == "swap_adjacent_chars":
        return swap_adjacent_chars(word)
    if method == "truncate_word":
        return truncate_word(word)
    raise ValueError(f"Unknown corruption method: {method}")


def corrupt_prompt(prompt: str, method: str) -> tuple[str, str]:
    """
    Corrupt one eligible word in the prompt.
    Returns:
        corrupted_prompt, original_word
    """
    words = prompt.split()
    word_idx = choose_word_index(words)

    if word_idx is None:
        return prompt, ""

    original_word = words[word_idx]
    corrupted = corrupt_word(original_word, method)
    words[word_idx] = corrupted
    return " ".join(words), original_word


VOWELS = "aeiou"
CONSONANTS = "bcdfghjklmnpqrstvwxyz"

def generate_pseudoword(length: int = None) -> str:
    """
    Generate a word-like non-word using alternating consonant/vowel patterns.
    """
    if length is None:
        length = random.randint(4, 7)

    # force minimum length
    length = max(length, 4)

    word = []
    for i in range(length):
        if i % 2 == 0:
            word.append(random.choice(CONSONANTS))
        else:
            word.append(random.choice(VOWELS))

    return "".join(word)

def replace_with_pseudoword(prompt: str) -> tuple[str, str, str]:
    """
    Replace one eligible content word in the prompt with a generated pseudoword.

    Returns:
        new_prompt, original_word, pseudoword
    """
    words = prompt.split()
    word_idx = choose_word_index(words)

    if word_idx is None:
        return prompt, "", ""

    original_word = words[word_idx]
    pseudoword = generate_pseudoword(max(len(original_word), 4))

    # avoid accidental identity, just in case
    while pseudoword.lower() == original_word.lower():
        pseudoword = generate_pseudoword(len(original_word))

    words[word_idx] = pseudoword
    new_prompt = " ".join(words)

    return new_prompt, original_word, pseudoword

def make_word_salad(prompt: str, max_tries: int = 50) -> tuple[str, list[str]]:
    """
    Shuffle the words of a prompt to create a stronger word-salad version.
    Keeps all words real, but rejects shuffles that still look too normal.
    """
    words = prompt.split()

    if len(words) < 3:
        return prompt, words

    banned_starts = {
        "a photo",
        "an photo",
        "a red",
        "a blue",
        "a green",
        "a yellow",
        "a black",
        "a white",
        "a shiny",
        "a dirty",
        "a clean",
        "a small",
        "a large",
        "two dogs",
        "two cats",
    }

    banned_last_words = {"a", "an"}

    for _ in range(max_tries):
        shuffled = words[:]
        random.shuffle(shuffled)

        if shuffled == words:
            continue

        shuffled_prompt = " ".join(shuffled)

        # reject weak shuffles that still look too grammatical
        first_two = " ".join(shuffled[:2]).lower()
        last_word = shuffled[-1].lower()

        if first_two in banned_starts:
            continue

        if last_word in banned_last_words:
            continue

        return shuffled_prompt, words

    return prompt, words

def generate_gibberish_token(min_len: int = 4, max_len: int = 8) -> str:
    """
    Generate a gibberish token that is not word-like.
    Uses random consonant-heavy sequences.
    """
    alphabet = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"

    length = random.randint(min_len, max_len)

    token = []

    for i in range(length):
        if random.random() < 0.7:
            token.append(random.choice(alphabet))
        else:
            token.append(random.choice(vowels))

    return "".join(token)