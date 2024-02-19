from pickle import load as pkl_load
from numpy import dot
from numpy.linalg import norm
from openai import OpenAI


with open("resources/api_key", "r") as f:
    api_key = f.read().strip()

with open("resources/org_id", "r") as f:
    org_id = f.read().strip()

client = OpenAI(api_key=api_key, organization=org_id)

with open("resources/wordlist-ada.pkl", "rb") as f:
    word_vecs: dict[str, list[float]] = pkl_load(f)


def get_similar_words(text: list[str], negative: list[str], n: int = 8):
    """
    Get the n most similar words to the input word
    """
    DIMENSIONS = 3072
    MODEL = "text-embedding-ada-002"  # "text-embedding-3-large"

    vecs = [
        i.embedding
        for i in client.embeddings.create(
            input=text, model=MODEL  # , dimensions=DIMENSIONS
        ).data
    ]

    neg_vecs = [
        i.embedding
        for i in client.embeddings.create(
            input=negative, model=MODEL  # , dimensions=DIMENSIONS
        ).data
    ]

    # find words that minimize the distance between the input word and the word in the wordlist
    sort = sorted(
        word_vecs.keys(),
        key=lambda word: sum(
            dot(vec, word_vecs[word]) / (norm(vec) * norm(word_vecs[word]))
            for vec in vecs
        )
        - sum(
            dot(vec, word_vecs[word]) / (norm(vec) * norm(word_vecs[word]))
            for vec in neg_vecs
        ),
        reverse=True,
    )

    out = []
    for word in sort:
        if any(i in word for i in text):
            continue
        out.append(word)
        if len(out) >= n:
            break
    return out


if __name__ == "__main__":
    print(get_similar_words(["track", "bolt", "dwarf"], ["mechanical", "bullet"]))
