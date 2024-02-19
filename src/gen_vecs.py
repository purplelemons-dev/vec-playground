if __name__ == "__main__":
    from openai import OpenAI
    import pickle as pkl
    from numpy.linalg import norm

    with open("resources/api_key", "r") as f:
        api_key = f.read().strip()

    with open("resources/org_id", "r") as f:
        org_id = f.read().strip()

    with open("resources/wordlist.txt", "r") as f:
        wordlist = [i.strip() for i in f.readlines()]

    client = OpenAI(api_key=api_key, organization=org_id)

    db = {}
    CHUNKSIZE = 1536

    for idx in range(0, len(wordlist), CHUNKSIZE):
        print(
            f"Processing chunk {idx // CHUNKSIZE + 1} of {len(wordlist) // CHUNKSIZE}"
        )
        chunk = wordlist[idx : idx + CHUNKSIZE]
        for word, embed_obj in zip(
            chunk,
            client.embeddings.create(
                input=chunk, model="text-embedding-ada-002"#, dimensions=3072
            ).data,
        ):
            n = norm(embed_obj.embedding)
            db[word] = embed_obj.embedding / n if n != 0 else embed_obj.embedding

    with open("resources/wordlist-ada.pkl", "wb") as f:
        pkl.dump(db, f, protocol=pkl.HIGHEST_PROTOCOL)
