It is a very basic N-gram Model for TEXT.

How to run:

1. Make sure to install these 2 libraries:  
NLTK, https://www.nltk.org/  
NLTK corpora, https://www.nltk.org/data.html

2. This code runs on Python3.

3. Arguments:  
The first argument takes in the corpus file. (The default setting is shakespeare.txt, warpiece.txt is also available in the repository)  
The second argument takes in the delta value for smoothing.  
The third argument takes in the random seed for text generation.  
```ruby
parser.add_argument('corpus_path', nargs="?", type=str, default='shakespeare.txt', help='Path to corpus file')
parser.add_argument('delta', nargs="?", type=float, default=.0, help='Delta value used for smoothing')
parser.add_argument('seed', nargs="?", type=int, default=82761904, help='Random seed used for text generation')
```
You can modify these 3 arguments as you wish.

4. Test Case of Sentence Log Probability:
```ruby
s1 = 'God has given it to me, let him who touches it beware!'
s2 = 'Where is the prince, my Dauphin?'
```
You can modify the test case as you wish.

4. Download Ngram.py and the corpus file, put them in the same directory.
5. Go to terminal -> set the directory to the address where Ngram.py and corpus file resides.
6. Run 
```
$ python3 N-gram.py
```

The result will show:  
Sentence log probability for s1 (Double)  
Sentence log probability for s2 (Double)  
5 random generated sentences from the corpus (String)  
