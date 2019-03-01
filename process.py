import pandas as pd
import json
from collections import defaultdict

df = pd.read_csv('data/train.csv')
d = {}
word_size = 49998
max_l = 0

dl = {}

for l in df['comment_text']:
  l = l.replace('"', '').replace('.', '').split()
  sl = len(l)
  max_l = max(max_l, sl)
  if sl in dl:
    dl[sl] += 1
  else:
    dl[sl] = 1

  for w in l:
    if w in d:
      d[w] += 1
    else:
      d[w] = 1

d = sorted(d.items(), key=lambda x: x[1], reverse=True)
d2 = {}

d2 = {'__EOS__' : 0, '__UNK__' : 1}

wf = open('data/word', 'w')

for i in range(word_size):
  d2[d[i][0]] = i + 2
  wf.write(d[i][0] + '\n')

wf.close()

json.dump(d2, open('data/dict', 'w'))
print max_l

print "Total Sentence : {} ".format(len(df))

cnt = 0
cnts = []

for i in dl:
  cnts.append([i, cnt + dl[i]])
  cnt += dl[i]

print cnts[50:70]

