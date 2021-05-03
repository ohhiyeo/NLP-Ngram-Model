import argparse
import math
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List
from typing import Tuple
from typing import Generator


# n is a (non-negative) int
# text is a list of strings
# Returns n-gram tuples of the form (string, context), where context is a tuple of strings
def get_ngrams(n: int, text: List[str]):
    newText =  text[:]
    newText = ['<s>']*(n-1)+newText+['</s>']
    ngrams = []
    for i, word in enumerate(newText):
        if word != '<s>':
            context = tuple(newText[i-n+1:i])
            ngrams.append((word, context))

    return ngrams

# Loads and tokenizes a corpus
# corpus_path is a string
# Returns a list of sentences, where each sentence is a list of strings
def load_corpus(corpus_path: str) -> List[List[str]]:

    t = open(corpus_path, 'r')
    paragraph = t.read().split('\n\n')
    sentTokenize = []
    for p in paragraph:
        for sent in sent_tokenize(p):
            sentTokenize.append(word_tokenize(sent))

    return sentTokenize


# Builds an n-gram model from a corpus
# n is a (non-negative) int
# corpus_path is a string
# Returns an NGramLM
def create_ngram_lm(n: int, corpus_path: str) -> 'NGramLM':
    train = NGramLM(n)
    wordText = load_corpus(corpus_path)
    for s in wordText:
        train.update(s)

    return train

# An n-gram language model
class NGramLM:
    def __init__(self, n: int):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocabulary = set()

    # Updates internal counts based on the n-grams in text
    # text is a list of strings
    # No return value
    def update(self, text: List[str]) -> None:

        contextTuple = get_ngrams(self.n,text)
        #store the word in self.vocabulary
        for word in contextTuple:
            self.vocabulary.add(word[0])
            if word in self.ngram_counts:
                self.ngram_counts[word] += 1
            else:
                self.ngram_counts[word] = self.ngram_counts.get(word,0)+1

            if word[1] in self.context_counts:
                self.context_counts[word[1]] += 1
            else:
                self.context_counts[word[1]] = self.context_counts.get(word[1],0)+1



    # Calculates the MLE probability of an n-gram
    # word is a string
    # context is a tuple of strings
    # delta is an float
    # Returns a float
    def get_ngram_prob(self, word: str, context: Tuple[str, ...], delta= .0) -> float:
        wcTuple = (word,context)
        # print(word, context)
        prob = 0
        if context not in self.context_counts:
            return (1/len(self.vocabulary))

        if wcTuple not in self.ngram_counts:
            return 0

        if delta == 0:
            prob = ((self.ngram_counts[wcTuple]+delta)/(self.context_counts[context]+delta*len(self.vocabulary)))
        else:
            prob = ((self.ngram_counts[wcTuple]+delta)/(self.context_counts[context]+delta*len(self.vocabulary)))
        # print(prob)
        return prob

    # Calculates the log probability of a sentence
    # sent is a list of strings
    # delta is a float
    # Returns a float
    def get_sent_log_prob(self, sent: List[str], delta=.0) -> float:
        sentTuple = list(get_ngrams(self.n, sent))

        sentProb = 0
        for sT in sentTuple:
            word = sT[0]
            if self.n == 1:
                context = ()
            else:
                context = sT[1]
            returnProb = self.get_ngram_prob(word,context)
            if returnProb == 0:
                return -math.inf
            wordProb = math.log2(returnProb)
            sentProb = sentProb+wordProb

        return sentProb

    # Calculates the perplexity of a language model on a test corpus
    # corpus is a list of lists of strings
    # Returns a float
    def get_perplexity(self, corpus: List[List[str]]) -> float:
        l = 0
        for cs in corpus:
            p = self.get_sent_log_prob(cs)
            l = l+p

        l = (l*(1/len(self.vocabulary)))
        perplex = math.pow(2,-l)

        return perplex

    # Samples a word from the probability distribution for a given context
    # context is a tuple of strings
    # delta is an float
    # Returns a string
    def generate_random_word(self, context: Tuple[str, ...], delta=.0) -> str:
        sortedVocab = sorted(self.vocabulary)
        r = random.random()
        previousSum = 0
        maxSum = 0
        wordReturn = ''
        for vocab in sortedVocab:
            prob = self.get_ngram_prob(vocab, context, delta)
            maxSum = previousSum + prob
            # wordList[vocab] = maxSum
            if maxSum > r and previousSum <= r :
                wordReturn = vocab
            previousSum = maxSum
        return wordReturn

    # Generates a random sentence
    # max_length is an int
    # delta is a float
    # Returns a string
    def generate_random_text(self, max_length: int, delta=.0) -> str:
        startContext = ('<s>',)*(self.n-1)
        preContext = ()
        newContext = ''
        sentence = []
        for i in range(max_length):
            if i == 0 and self.n!=1:
                preContext = startContext
            elif i != 0 and self.n!=1:
                preContext = list(preContext)
                preContext.pop(0)
                preContext.append(newContext)
                preContext = tuple(preContext)
            word = self.generate_random_word(preContext, delta)
            newContext = word
            sentence.append(word)
            if word == '</s>':
                break

        return ' '.join(sentence)

def main(corpus_path: str, delta: float, seed: int):
    trigram_lm = create_ngram_lm(1, corpus_path)
    s1 = 'God has given it to me, let him who touches it beware!'
    s2 = 'Where is the prince, my Dauphin?'
    #
    print(trigram_lm.get_sent_log_prob(word_tokenize(s1)))
    print(trigram_lm.get_sent_log_prob(word_tokenize(s2)))
    for i in range(5):
        print(trigram_lm.generate_random_text(10))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="N-gram Method")
    parser.add_argument('corpus_path', nargs="?", type=str, default='shakespeare.txt', help='Path to corpus file')
    parser.add_argument('delta', nargs="?", type=float, default=.0, help='Delta value used for smoothing')
    parser.add_argument('seed', nargs="?", type=int, default=82761904, help='Random seed used for text generation')
    args = parser.parse_args()
    random.seed(args.seed)
    main(args.corpus_path, args.delta, args.seed)
