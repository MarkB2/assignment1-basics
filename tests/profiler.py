import cProfile
import pstats
import tokenizer

with cProfile.Profile() as pr:
    merges = tokenizer.train('tests/fixtures/corpus.en', vocab_size=500)

stats = pstats.Stats(pr)
stats.sort_stats('cumulative')   # or 'tottime', 'ncalls'
stats.print_stats(20)            # top 20 lines