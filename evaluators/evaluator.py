from inspect import getmembers, isfunction

from . import metrics

metric_fn_dict = dict(getmembers(metrics, predicate=isfunction))


class Evaluator:
    def __init__(self, model, *,
                 valid_loader, test_loader,
                 valid_metrics=None,
                 test_metrics=None,
                 **kwargs):
        self.model = model
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.valid_metrics = valid_metrics or {}
        self.test_metrics = test_metrics or valid_metrics
        self.metric_kwargs = kwargs

    def evaluate(self, dataloader, metric, unprivileged_group=0.5, privileged_group=0.5):
        assert metric in metric_fn_dict, f"Metric name {metric} not present in `metrics.py`"

        metric_fn = metric_fn_dict[metric]

        self.model.eval()
        return metric_fn(self.model, dataloader, unprivileged_group, privileged_group, **self.metric_kwargs)

    def validate(self):
        print(f"Validating {self.valid_metrics}")
        return {metric: self.evaluate(self.valid_loader, metric)
                for metric in self.valid_metrics}

    def test(self):
        print(f"Testing {self.test_metrics}")
        results = {}
        unprivileged_group = 0.5
        privileged_group = 0.5
        if "choose_thresholds" in self.test_metrics:
            data_loader = [self.valid_loader, self.test_loader]
            results["choose_thresholds"] = self.evaluate(data_loader, "choose_thresholds",
                                                         unprivileged_group, privileged_group)
            unprivileged_group, privileged_group = results["choose_thresholds"]
            self.test_metrics.remove("choose_thresholds")
        if "bayes_thresholds" in self.test_metrics:
            data_loader = [self.valid_loader, self.test_loader]
            results["bayes_thresholds"] = self.evaluate(data_loader, "bayes_thresholds",
                                                         unprivileged_group, privileged_group)
            unprivileged_group, privileged_group = results["bayes_thresholds"]
            self.test_metrics.remove("bayes_thresholds")

        for metric in self.test_metrics:
            data_loader = self.test_loader
            results[metric] = self.evaluate(data_loader, metric, unprivileged_group, privileged_group)
        return results


def create_evaluator(model, valid_loader, test_loader, valid_metrics, test_metrics, **kwargs):
    valid_metrics = set(valid_metrics)
    test_metrics = set(test_metrics)

    return Evaluator(
        model,
        valid_loader=valid_loader,
        test_loader=test_loader,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        **kwargs
    )
