#!/usr/bin/env python

from collections import defaultdict
from itertools import chain, combinations, product


class SpellingCorrector(object):

    def __init__(self, dictionary):
        self.load_dictionary(dictionary)

    def load_dictionary(self, filename):
        self.dic = set()
        with open(filename) as fp:
            for line in fp:
                self.dic.add(line[:-1].lower())

    def correct_phrase(self, word_list):
        word_list = [str(w) for w in word_list]  # Temp hack
        word_cache = defaultdict(float)
        phrase_score = 0.
        phrase_winner = word_list
        for phrase in powerset_join(word_list):
            tweaked_phrase = []
            for word in phrase:
                word_score = 0.
                word_winner = word
                for ntweaks, tweak in self.transforms(word):
                    if tweak not in word_cache:
                        word_cache[tweak] = self.score_word(tweak)
                    score = word_cache[tweak] * len(word) / (len(word) + ntweaks)
                    if word_score < score:
                        word_score = score
                        word_winner = tweak
                tweaked_phrase.append(word_winner)
            score = sum(word_cache[w] for w in tweaked_phrase) / len(tweaked_phrase)
            if phrase_score < score:
                phrase_score = score
                phrase_winner = tweaked_phrase

        return phrase_winner

    def transforms(self, word):
        inserts = "IJTLPH"

        if self.check(word):
            yield 0, word

        pos = range(len(word) + 1)
        for i in pos:
            for c in inserts:
                t = word[:i] + c + word[i:]
                if self.check(t):
                    yield 1, t

        for i, j in combinations(pos, 2):
            for a, b in product(inserts, repeat=2):
                t = word[:i] + a + word[i:j] + b + word[j:]
                if self.check(t):
                    yield 2, t

    def check(self, word):
        return word.lower() in self.dic

    def score_word(self, word):
        return len(word)


def powerset_join(words):
    """['a', 'b', 'c'] -> [['a', 'b', 'c'], ['ab', 'c'], ['a', 'bc'], ['abc']]"""

    space_pos = range(len(words) - 1)
    for combo in chain.from_iterable(combinations(space_pos, r) for r in range(len(words))):
        phrase = [words[0]]
        for i in range(len(words) - 1):
            if i in combo:
                phrase[-1] += words[i + 1]
            else:
                phrase.append(words[i + 1])
        yield phrase


if __name__ == '__main__':
    import sys
    sc = SpellingCorrector('/usr/share/dict/words')
    print sc.correct_phrase(sys.argv[1:])
