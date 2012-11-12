
import os

import cv


class Classifications(object):

    def certainty(self, char):
        pass

    def rankings(self, limit=None, ignore_case=True):
        pass

    def filter_rankings(self, r, limit=None, ignore_case=True):
        if ignore_case:
            seen = set()
            r = [c for c in r if c not in seen and not seen.add(c)]
        if limit is not None:
            r = r[:limit]
        return r

def score_files(files, test, ignore_case=True):
    correct = 0
    min_correct_certainty = 1
    ave_correct_certainty = 0
    max_incorrect_certainty = 0
    ave_incorrect_certainty = 0
    incorrect = []
    for f in files:
        char = os.path.basename(f)[0]  # Assume filename starts with char
        i = cv.LoadImageM(f, cv.CV_LOAD_IMAGE_GRAYSCALE)
        classifications = test(i, char=char)
        rankings = classifications.rankings(limit=4)
        guess = rankings[0]
        cert = classifications.certainty(guess)
        case_char = char
        case_guess = guess
        if ignore_case:
            char = char.lower()
            guess = guess.lower()
        if guess == char:
            correct += 1
            min_correct_certainty = min(cert, min_correct_certainty)
            ave_correct_certainty += cert
        else:
            incorrect.append((case_char, case_guess, cert, rankings[1:]))
            max_incorrect_certainty = max(max_incorrect_certainty, cert)
            ave_incorrect_certainty += cert

    n = len(files)
    ave_correct_certainty = float(ave_correct_certainty) / correct
    ave_incorrect_certainty = float(ave_incorrect_certainty) / (n - correct)
    print 'Classified %d/%d correctly (%.2f%%)' % (correct, n, 100. * correct / n)
    print 'Average certainty for correct classification: %.3f (min: %.3f)' % (ave_correct_certainty, min_correct_certainty)
    print 'Average certainty for incorrect classification: %.3f (max: %.3f)' % (ave_incorrect_certainty, max_incorrect_certainty)
    print 'Incorrect classifications:'
    for c, g, r, alt in incorrect:
        print '\tThought %s was %s (certainty: %.3f); alternatives were %s' % (c, g, r, ', '.join(alt))
