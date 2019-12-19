from nltk.translate.bleu_score import corpus_bleu

"""
Evaluator class for computing BLEU
Args:
    metrics:       list containing compute metrics
"""
class BleuEval():
    def __init__(self, metrics):
        self.metrics = metrics

    """
    Compute metrics
    Args:
        ref_list:       list(list(str)) containing ground truth sentences
        hyp_list:       list(str) containing predicted sentences
    """
    def compute_metrics(self, ref_list, hyp_list):
        assert len(ref_list) == len(hyp_list)
        ref = [[sentence.split() for sentence in refs] for refs in ref_list]
        hyp = [sentence.split() for sentence in hyp_list]
        metric_dict = {}
        for metric in self.metrics:
            if metric.upper().startswith("BLEU"):
                k = float(metric[-1])
                weights = (1/k for _ in range(k))
                metric_dict[metric] = corpus_bleu(ref, hyp, weights)
            else:
                print("metric {} is not implemented".format(metric))
        return metric_dict


