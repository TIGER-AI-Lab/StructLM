# encoding=utf8

from metrics.spider.spider_exact_match import compute_exact_match_metric
from metrics.spider.spider_test_suite import compute_test_suite_metric
from metrics.sparc.interaction_scores import compute_interaction_metric
import glob

class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        # if self.args.seq2seq.target_with_db_id:
        #     # Remove database id from all predictions
        #     preds = [pred.split("|", 1)[-1].strip() for pred in preds]
        matching_paths = glob.glob("data/downloads/extracted/*/sparc/database")
        assert len(matching_paths) == 1
        db_dir = matching_paths[0]
        golds[0]['db_path'] = db_dir

        # fix bugs in the gold dataset.
        for i, each in enumerate(golds):
            if "> =" in each['query']:
                golds[i]['query'] = each['query'].replace("> =", ">=")
            if "< =" in each['query']:
                golds[i]['query'] = each['query'].replace("< =", "<=") 
            if each['query'] == "SELECT countryname FROM countries WHERE countryid = 1 or countryid = 2 or countryid = 3 ) ":
                golds[i]['query'] = "SELECT countryname FROM countries WHERE countryid = 1 or countryid = 2 or countryid = 3"


        exact_match = compute_exact_match_metric(preds, golds)
        test_suite = compute_test_suite_metric(preds, golds, db_dir=self.args.test_suite_db_dir)
        if section in ["train", "dev"]:
            return {**exact_match, **test_suite}
        elif section == "test":
            interaction_scores = compute_interaction_metric(preds, golds)
            return {**exact_match, **test_suite, **interaction_scores}
