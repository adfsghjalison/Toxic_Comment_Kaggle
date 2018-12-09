import pandas as pd
import json

df = pd.read_csv('data/train.csv')
d = {}
word_size = 49998

for l in df['comment_text']:
  l = l.replace('"', '').replace('.', '').split()
  for w in l:
    if w in d:
      d[w] += 1
    else:
      d[w] = 1

d = sorted(d.items(), key=lambda x: x[1], reverse=True)
d2 = {}

d2 = {'__EOS__' : 0, '__UNK__' : 1}

for i in range(word_size):
  d2[d[i][0]] = i + 2

json.dump(d2, open('data/word.json', 'w'))

