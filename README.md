
# Vec-Playground

install with
```bash
git clone https://github.com/purplelemons-dev/vec-playground.git
```

after that, you need to get the `word:vec` dictionary, which can either be generated or downloaded. The vectors are *PRE-NORMALIZED*, so you don't need to normalize them yourself.

## Generating the wordlist
* Slow internet
* OpenAI API key
If you would like to generate the wordlist for yourself, use:
```bash
cd vec-playground
pip install --upgrade openai
echo "<YOUR OPENAI API KEY>" > resources/api_key
echo "<YOUR OPENAI ORG ID>" > resources/org_id
python gen_vecs.py
```

## Downloading the wordlist
* Fast internet
* Don't want to pay for OpenAI API
otherwise, you can download it (~1.7GB) from `https://purplelemons.s3.amazonaws.com/wordlist.plk` (NOTE: DO NOT OPEN IN BROWSER, YOU'RE GONNA NUKE YOUR PC).
place the `wordlist.plk` file in the `resources` directory.
