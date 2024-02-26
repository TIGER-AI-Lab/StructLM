# encoding=utf8
import numpy as np
import nltk

class EvaluateTool(object):

    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):

        full_hyps = [pred.lower().split() for pred in preds]
        # we gonna hack this for now. Since USKG only supports seq2seq, we will combine the golds into one string
        references = [item['sentences'] for item in golds]

        bleu_1 = nltk.translate.bleu_score.corpus_bleu(references, full_hyps, weights=(1.0, 0, 0))
        bleu_2 = nltk.translate.bleu_score.corpus_bleu(references, full_hyps, weights=(0.5, 0.5, 0))
        bleu_3 = nltk.translate.bleu_score.corpus_bleu(references, full_hyps, weights=(0.33, 0.33, 0.33))
        bleu_4 = nltk.translate.bleu_score.corpus_bleu(references, full_hyps, weights=(0.25, 0.25, 0.25, 0.25))

        print("Corpus BLEU: {}/{}/{}/{}".format(bleu_1, bleu_2, bleu_3, bleu_4))
        return {'bleu_1': bleu_1, 'bleu_2': bleu_2, 'bleu_3': bleu_3, 'bleu_4': bleu_4}
