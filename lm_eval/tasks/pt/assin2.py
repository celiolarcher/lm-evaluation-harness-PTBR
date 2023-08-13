# TODO: colocar description_dict ou mudar o few_show_context

# O QUE EU QUERIA:
# ASSIN2_RTE_SINGLE_EXAMPLE_PROMPT ="""\
# ###
# Exemplo {n}:
# Texto: {text}
# Hipótese: {hypothesis}
# Predição: {entailment}"""

# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""
from lm_eval.base import rf, Task
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
import numpy as np
from lm_eval.metrics import mean
from lm_eval import utils
from collections import defaultdict
import math

# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""

def items_to_trues_and_preds(items):
    trues, preds = zip(*items)
    assert len(trues) == len(preds)
    trues, preds = np.array(trues), np.array(preds)
    return trues, preds


def compute_macro_f1(items):
    trues, preds = items_to_trues_and_preds(items)
    macro_f1 = f1_score(y_true=trues,
                        y_pred=preds,
                        average="macro",
                        labels=list(set(trues)))
    return macro_f1

def compute_person(items):
    trues, preds = items_to_trues_and_preds(items)
    person = pearsonr(trues, preds)[0]
    return person

def compute_mse(items):
    trues, preds = items_to_trues_and_preds(items)
    mse = ((trues - preds) ** 2).mean()
    return mse


def string_to_float(x):
    """Converts a string to a float, returning -1 if the conversion fails."""
    try:
        return float(x)
    except:
        return -1

class Assin2Base(Task):
    """Base class for ASSIN2 tasks.

    Modifications:
    - `few_show_examples` method allow to select the same number of examples
    for each label. This avoids bias towards the majority class.
    - `fewshot_context` method modified to also pass index of the each few shot
    example (to be used in the prompt) and an indicator if the doc is a few shot
    or a text document.
    
    """
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "assin2"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                # TODO: Return the training document generator from `self.dataset`.
                # If you need to process the data, `map` over the documents with
                # the custom processing function, `self._process_doc`. E.g.
                # `map(self._process_doc, self.dataset["validation"])`
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                self._training_docs = list(map(self._process_doc,
                                               self.dataset["train"]))
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["test"])

    @utils.positional_deprecated
    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param provide_description: bool
            Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print(
                "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
            )

        description = description + "\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                # raise NotImplementedError(
                #     "Assin2 has training docs, so this should never happen")
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs()
                        if self.has_validation_docs()
                        else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = (
                "\n\n".join(
                    [
                        self.doc_to_text(doc, n_example_fewshot=n) + self.doc_to_target(doc)
                        for n, doc in enumerate(fewshotex, 1)
                    ]
                )
                + "\n\n"
            )

        example = self.doc_to_text(doc)
        return description + labeled_examples + example
    
    def fewshot_examples(self, k, rnd):
        """Essa aqui precisa dar um jeito de passar a key"""
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())
        
        # check if we have a key to balance by
        # if not just return a random sample (default behavior)
        if "__fewshot_balance_key__" not in self._training_docs[0]:
            return rnd.sample(self._training_docs, k)
        
        # otherwise, group by keyfunc and sample from each group
        groups = defaultdict(list)
        for doc in self._training_docs:
            groups[doc['__fewshot_balance_key__']].append(doc)
        n_groups = len(groups)
        examples_per_group = math.ceil(k / n_groups)
        out = [
            item for group in groups.values()
            for item in rnd.sample(group, examples_per_group)]
        out = rnd.sample(out, k)
        return out


class Assin2RTE(Assin2Base):
    VERSION = 0

    def _process_doc(self, doc):
        # TODO: Process (detokenize, strip, replace etc.) each individual `doc`
        # with this function. You can map this across the docs in each available
        # dataset split. See the TODOs in `train_docs`, `validation_docs`, and
        # `test_docs` for snippets.
        # NOTE: DELETE THIS FUNCTION IF UNUSED.
        # CLASS_LABELS = ["None", "Entailment"]
        CLASS_LABELS = ["Sim", "Não"]
        doc['label_name'] = CLASS_LABELS[doc["entailment_judgment"]]
        doc['__fewshot_balance_key__'] = doc['label_name']
        return doc

    def doc_to_text(self, doc, n_example_fewshot=None):
        # TODO: Format the query prompt portion of the document example.
        # return "\n".join([
        #     "###",
        #     # f"Premissa: {doc['premise']}",
        #     # f"Hipótese: {doc['hypothesis']}",
        #     # f"Predição:"
        #     f"[A]: {doc['premise']}",
        #     f"[B]: {doc['hypothesis']}",
        #     f"Resposta:"
        # ])
    
        # return "\n".join([
        #     "###",
        #     f"Sabendo que '{doc['premise']}' é verdadeiro, podemos dizer que '{doc['hypothesis']}' é verdadeiro? Sim ou não",
        #     f"Resposta:"
        # ])
        ex_index = f"Exemplo {n_example_fewshot}:" if n_example_fewshot else ""
        return "\n".join([
            f"###\n{ex_index}",
            f"Sabendo que '{doc['premise']}', podemos dizer que '{doc['hypothesis']}'? Sim ou não?",
            f"Resposta:"
        ])

    def doc_to_target(self, doc):
        # TODO: Fill in the `target` ("gold answer") variable.
        # The prepended `" "` is required to space out the `doc_to_text` and
        # `doc_to_target` strings.
        target = doc['label_name']
        return " " + target

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        # TODO: Construct your language model requests with the request factory, `rf`,
        # and return them as an iterable.
        continuation = rf.greedy_until(ctx, {"until": ["\n", " "]})
        return continuation

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and the corresponding metric result as value
        # for the current `doc`.
        true = doc["label_name"]
        pred = results[0]
        return {"accuracy": (true == pred), "macro_f1": (true, pred)}

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and an aggregation function as value which
        # determines how to combine results from each document in the dataset.
        # Check `lm_eval.metrics` to find built-in aggregation functions.
        return {'accuracy': mean, 'macro_f1': compute_macro_f1}

    def higher_is_better(self):
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and a `bool` value determining whether or
        # not higher values of that metric are deemed better.
        return {'macro_f1': True, 'accuracy': True}

class Assin2STS(Task):
    VERSION = 0
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "assin2"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                # this will be called just once
                train_ds_filter = self.dataset["train"].filter(
                    lambda x: x['relatedness_score'] in [1,2,3,4,5])
                self._training_docs = list(train_ds_filter)
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["test"])

    def doc_to_text(self, doc):
        return "\n".join([
            "###",
            f"Sentença 1: {doc['premise']}",
            f"Sentença 2: {doc['hypothesis']}"
            f"Similaridade:"
        ])

    def doc_to_target(self, doc):
        # TODO: Fill in the `target` ("gold answer") variable.
        # The prepended `" "` is required to space out the `doc_to_text` and
        # `doc_to_target` strings.
        target = str(int(doc['relatedness_score']))
        return " " + target

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        # TODO: Construct your language model requests with the request factory, `rf`,
        # and return them as an iterable.
        continuation = rf.greedy_until(ctx, {"until": ["\n", " "]})
        return continuation

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and the corresponding metric result as value
        # for the current `doc`.
        true = doc["relatedness_score"]
        pred = string_to_float(results[0])
        return {"pearson": (true, pred), "mse": (true, pred)}

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and an aggregation function as value which
        # determines how to combine results from each document in the dataset.
        # Check `lm_eval.metrics` to find built-in aggregation functions.
        return {'pearson': compute_person, 'mse': compute_mse}

    def higher_is_better(self):
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and a `bool` value determining whether or
        # not higher values of that metric are deemed better.
        return {'pearson': True, 'mse': False}
