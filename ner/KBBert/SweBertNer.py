from transformers import pipeline

from spacy.tokens import Doc, Span, Token


class SweBertNer(object):

    def __init__(self, nlpTrans, nlp):
        self.nlpTrans = nlpTrans
        self.nlp = nlp

    def __call__(self, doc):
        sents = list(doc.sents)

        entitiesInSents = list(self.findEntitiesInSents(sents))

        entitiesTokenLists = list(self.toTokenLists(entitiesInSents))
        indicesForEntsDup = list(self.findIndicesForEnts(sents, entitiesTokenLists, doc))

        #Remove duplicates
        seen, indicesForEnts = set(), []
        for ele in indicesForEntsDup:
            tp = tuple(ele[0:len(ele) - 1])
            if tp not in seen:
                indicesForEnts.append(ele)
            seen.add(tp)

        #Create the entity spans and add them to doc.ents
        for indices in indicesForEnts:
            entity = Span(doc, indices[0], indices[len(indices) - 2] + 1, label=self.nlp.vocab.strings[indices[len(indices) - 1]])
            doc.ents = list(doc.ents) + [entity]
        return doc

    #returns a list of lists of indices for each entity, entity type in the last element

    def findIndicesForEnts(self, sents, entitiesTokenLists, doc):
        for sent, entSent in zip(sents, entitiesTokenLists):
            tokens = list(sent)
            stringTokens = [str(x) for x in tokens]
            for ent in entSent:
                entIndices = []
                cursor = 0
                numOfNewlines = 0
                for tok in tokens:
                    if tok.text == '\n' and cursor > 0:
                        numOfNewlines += 1
                        entIndices.append(tok.i)
                        continue
                    elif tok.text == ent[cursor]:
                        print(str(tok.i))
                        entIndices.append(tok.i)
                        cursor += 1
                        if cursor == len(ent) - 1:
                            entIndices.append(ent[-1])
                            yield entIndices
                            cursor = 0
                    else:
                        entIndices = []
                        cursor = 0

    #Run transformer ner on sentences in doc, returns a list of entity markups
    def findEntitiesInSents(self, sents):
        for sent in sents:
            tokens = list(sent)
            stringTokens = [str(x) for x in tokens]
            yield self.nlpTrans(' '.join(stringTokens))

    #Returns a list of tokens for each entity, appends entity type at the end each list
    def findTokens(self, ents):
        for i, ent in enumerate(ents):
            if ent['entity'].startswith('B'):
                entityLabel = ent['entity'].split('-')[1]
                startIndex = i
                numOfParts = 1
                for ent2 in ents[i + 1:]:
                    if ent2['entity'].startswith('I'):
                        numOfParts += 1
                    else:
                        break
                str = ''
                for x in range(startIndex, startIndex + numOfParts):
                    str += (ents[x]['word'] + ' ')
                strConcat = str.replace(' ##', '').strip()
                entList = strConcat.split()
                entList.append(entityLabel)
                yield entList

    def toTokenLists(self, entitiesInSents):
        for ents in entitiesInSents:
            startIndexes = list(self.findTokens(ents))
            yield startIndexes
