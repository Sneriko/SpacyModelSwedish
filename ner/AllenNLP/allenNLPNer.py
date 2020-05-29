import spacy
from allennlp.predictors import sentence_tagger
from allennlp.models.archival import load_archive
import allennlp_models.ner.crf_tagger
from allennlp.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from spacy.tokens import Doc, Span, Token



class AllenNLPNer(object):

    def __init__(self, predictor):
        self.predictor = predictor

    def __call__(self, doc):
        sents = list(doc.sents)
        self.predictor._tokenizer = WhitespaceTokenizer()
        for sent in sents:
            startIndex = sent.start
            tokens = list(sent)
            stringTokens = [str(x) for x in tokens]
            allenObj = self.predictor.predict(sentence=" ".join(stringTokens))
            new_line_indicies = [i for i, x in enumerate(stringTokens) if x.strip(' ') == "\n" or x.strip(' ') == '\xa0']
            if len(stringTokens) != len(allenObj["words"]) and len(new_line_indicies) > 0:
                for i in new_line_indicies:
                    allenObj["words"].insert(i, stringTokens[i])
                    allenObj["tags"].insert(i, "O")
            if len(tokens) != len(allenObj['words']):
                print(str(len(tokens)) + ' ' + str(len(allenObj['words'])))
                raise Exception('tokenizer error')
            for i, (tag, word, token) in enumerate(zip(allenObj["tags"], allenObj["words"], tokens)):
                if tag.startswith('U'):
                    tagSplitted = tag.split('-')
                    entity = Span(doc, sent.start + i, sent.start + i + 1, label=tagSplitted[1])
                    doc.ents = list(doc.ents) + [entity]
                    continue
                elif tag.startswith('B'):
                    startIndex = sent.start + i
                    continue
                elif tag.startswith('L'):
                    tagSplitted = tag.split('-')
                    entity = Span(doc, startIndex, sent.start + i + 1, label=tagSplitted[1])
                    doc.ents = list(doc.ents) + [entity]
                else:
                    continue
        return doc
