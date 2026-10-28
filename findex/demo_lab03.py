import sys

from findex.ranked_search import main

sys.argv = [
    "ranked_search",
    "index.bin",
    '"пошукова система" OR (python AND київ) NOT java',
    "--corpus",
    "data/",
    "--top",
    "5",
]

if __name__ == "__main__":
    main()