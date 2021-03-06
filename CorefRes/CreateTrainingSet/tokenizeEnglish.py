import io
import os
import spacy

nlp = spacy.load('en_core_web_md', disable=['tagger', 'parser', 'ner'])
numOfTokens = 0
with io.open('../Data/Datasets/Europarl/Documents/ep-09-03-11-018/tokenized/ep-09-03-11-018-en-tok.txt', 'w+', encoding='utf8') as fo:
    with io.open('../Data/Datasets/Europarl/Documents/ep-09-03-11-018/text/ep-09-03-11-018-en.txt', encoding='utf8')as fi:
        for line in fi:
            doc = nlp(line)
            for token in doc:
                numOfTokens = numOfTokens + 1
                if token.text != '\n':
                    fo.write(token.text + ' ')
                else:
                    fo.write(token.text)
