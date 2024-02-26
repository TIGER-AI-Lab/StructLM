# encoding=utf8

from metrics.spider.spider_exact_match import compute_exact_match_metric
from metrics.spider.spider_test_suite import compute_test_suite_metric
from metrics.sparc.interaction_scores import compute_interaction_metric
import glob


class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):

        matching_paths = glob.glob("data/downloads/extracted/*/sparc/database")
        assert len(matching_paths) == 1
        db_dir = matching_paths[0]
        golds[0]['db_path'] = db_dir

        if self.args.seq2seq.target_with_db_id:
            # Remove database id from all predictions
            preds = [pred.split("|", 1)[-1].strip() for pred in preds]

        for i in range(len(golds)):
            if golds[i]['query'].lower() == "SELECT T1.id, T1.name FROM battle EXCEPT SELECT T1.id, T1.name FROM battle AS T1 JOIN ship AS T2 ON T1.id  =  T2.lost_in_battle WHERE T2.location  =  'English Channel'".lower():
                golds[i]['query'] = "SELECT id, name FROM battle EXCEPT SELECT T1.id, T1.name FROM battle AS T1 JOIN ship AS T2 ON T1.id  =  T2.lost_in_battle WHERE T2.location  =  'English Channel'"

        exact_match = compute_exact_match_metric(preds, golds)
        test_suite = compute_test_suite_metric(preds, golds, db_dir=db_dir)
        if section in ["train", "dev"]:
            return {**exact_match, **test_suite}
        elif section == "test":
            interaction_scores = compute_interaction_metric(preds, golds)
            return {**exact_match, **test_suite, **interaction_scores}
