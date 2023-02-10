import numpy as np
import torch
import heapq


class AbstractMetric(object):
    """:class:`AbstractMetric` is the base object of all metrics. If you want to
        implement a metric, you should inherit this class.
    Args:
        config (Config): the config of evaluator.
    """

    smaller = False

    def __init__(self, config):
        self.decimal_place = config["metric_decimal_place"]

    def calculate_metric(self, dataobject):
        """Get the dictionary of a metric.
        Args:
            dataobject(DataStruct): it contains all the information needed to calculate metrics.
        Returns:
            dict: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
        """
        raise NotImplementedError("Method [calculate_metric] should be implemented.")


class TopkMetric(AbstractMetric):
    """:class:`TopkMetric` is a base object of top-k metrics. If you want to
    implement an top-k metric, you can inherit this class.
    Args:
        config (Config): The config of evaluator.
    """

    # metric_type = EvaluatorType.RANKING
    # metric_need = ["rec.topk"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]

    def used_info(self, dataobject):
        """Get the bool matrix indicating whether the corresponding item is positive
        and number of positive items for each user.
        """
        rec_mat = dataobject.get("rec.topk")
        topk_idx, pos_len_list = torch.split(rec_mat, [max(self.topk), 1], dim=1)
        return topk_idx.to(torch.bool).numpy(), pos_len_list.squeeze(-1).numpy()

    def topk_result(self, metric, value):
        """Match the metric value to the `k` and put them in `dictionary` form.
        Args:
            metric(str): the name of calculated metric.
            value(numpy.ndarray): metrics for each query, including values from `metric@1` to `metric@max(self.topk)`.
        Returns:
            dict: metric values required in the configuration.
        """
        metric_dict = {}
        avg_result = value.mean(axis=0)
        for k in self.topk:
            key = "{}@{}".format(metric, k)
            metric_dict[key] = round(avg_result[k - 1], self.decimal_place)
        return metric_dict

    def metric_info(self, pos_index, pos_len=None):
        """Calculate the value of the metric.
        Args:
            pos_index(numpy.ndarray): a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th \
            highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
            pos_len(numpy.ndarray): a vector representing the number of positive items per user, shape of ``(n_users,)``.
        Returns:
            numpy.ndarray: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
        """
        raise NotImplementedError(
            "Method [metric_info] of top-k metric should be implemented."
        )


class MRR(TopkMetric):
    r"""The MRR_ (also known as Mean Reciprocal Rank) computes the reciprocal rank
    of the first relevant item found by an algorithm.
    .. _MRR: https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    .. math::
       \mathrm {MRR@K} = \frac{1}{|U|}\sum_{u \in U} \frac{1}{\operatorname{rank}_{u}^{*}}
    :math:`{rank}_{u}^{*}` is the rank position of the first relevant item found by an algorithm for a user :math:`u`.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, _ = self.used_info(dataobject)
        result = self.metric_info(pos_index)
        metric_dict = self.topk_result("mrr", result)
        return metric_dict

    def metric_info(self, pos_index):
        idxs = pos_index.argmax(axis=1)
        result = np.zeros_like(pos_index, dtype=np.float)
        for row, idx in enumerate(idxs):
            if pos_index[row, idx] > 0:
                result[row, idx:] = 1 / (idx + 1)
            else:
                result[row, idx:] = 0
        return result


class Recall(TopkMetric):
    r"""Recall_ is a measure for computing the fraction of relevant items out of all relevant items.
    .. _recall: https://en.wikipedia.org/wiki/Precision_and_recall#Recall
    .. math::
       \mathrm {Recall@K} = \frac{1}{|U|}\sum_{u \in U} \frac{|\hat{R}(u) \cap R(u)|}{|R(u)|}
    :math:`|R(u)|` represents the item count of :math:`R(u)`.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result("recall", result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
        return np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)


class NDCG(TopkMetric):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality,
    where positions are discounted logarithmically. It accounts for the position of the hit by assigning
    higher scores to hits at top ranks.
    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
    .. math::
        \mathrm {NDCG@K} = \frac{1}{|U|}\sum_{u \in U} (\frac{1}{\sum_{i=1}^{\min (|R(u)|, K)}
        \frac{1}{\log _{2}(i+1)}} \sum_{i=1}^{K} \delta(i \in R(u)) \frac{1}{\log _{2}(i+1)})
    :math:`\delta(·)` is an indicator function.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result("ndcg", result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

        iranks = np.zeros_like(pos_index, dtype=np.float)
        iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
        for row, idx in enumerate(idcg_len):
            idcg[row, idx:] = idcg[row, idx - 1]

        ranks = np.zeros_like(pos_index, dtype=np.float)
        ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        dcg = 1.0 / np.log2(ranks + 1)
        dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

        result = dcg / idcg
        return result


def mrr_(pos_index, pos_len):
    r"""The MRR_ (also known as mean reciprocal rank) is a statistic measure for evaluating any process
    that produces a list of possible responses to a sample of queries, ordered by probability of correctness.

    .. _MRR: https://en.wikipedia.org/wiki/Mean_reciprocal_rank

    .. math::
        \mathrm {MRR} = \frac{1}{|{U}|} \sum_{i=1}^{|{U}|} \frac{1}{rank_i}

    :math:`U` is the number of users, :math:`rank_i` is the rank of the first item in the recommendation list
    in the test set results for user :math:`i`.

    """
    idxs = pos_index.argmax(axis=1)
    result = np.zeros_like(pos_index, dtype=np.float)
    for row, idx in enumerate(idxs):
        if pos_index[row, idx] > 0:
            result[row, idx:] = 1 / (idx + 1)
        else:
            result[row, idx:] = 0
    return result


def recall_(pos_index, pos_len):
    r"""Recall_ (also known as sensitivity) is the fraction of the total amount of relevant instances
    that were actually retrieved

    .. _recall: https://en.wikipedia.org/wiki/Precision_and_recall#Recall

    .. math::
        \mathrm {Recall@K} = \frac{|Rel_u\cap Rec_u|}{Rel_u}

    :math:`Rel_u` is the set of items relevant to user :math:`U`,
    :math:`Rec_u` is the top K items recommended to users.
    We obtain the result by calculating the average :math:`Recall@K` of each user.

    """
    return np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)


def ndcg_(pos_index, pos_len):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality.
    Through normalizing the score, users and their recommendation list results in the whole test set can be evaluated.

    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    .. math::
        \begin{gather}
            \mathrm {DCG@K}=\sum_{i=1}^{K} \frac{2^{rel_i}-1}{\log_{2}{(i+1)}}\\
            \mathrm {IDCG@K}=\sum_{i=1}^{K}\frac{1}{\log_{2}{(i+1)}}\\
            \mathrm {NDCG_u@K}=\frac{DCG_u@K}{IDCG_u@K}\\
            \mathrm {NDCG@K}=\frac{\sum \nolimits_{u \in U^{te}NDCG_u@K}}{|U^{te}|}
        \end{gather}

    :math:`K` stands for recommending :math:`K` items.
    And the :math:`rel_i` is the relevance of the item in position :math:`i` in the recommendation list.
    :math:`{rel_i}` equals to 1 if the item is ground truth otherwise 0.
    :math:`U^{te}` stands for all users in the test set.

    """
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index, dtype=np.float)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=np.float)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    return result


def topk_result(metric, value, topk, decimal_place):
    """Match the metric value to the `k` and put them in `dictionary` form.
    Args:
        metric(str): the name of calculated metric.
        value(numpy.ndarray): metrics for each query, including values from `metric@1` to `metric@max(self.topk)`.
    Returns:
        dict: metric values required in the configuration.
    """
    metric_dict = {}
    avg_result = value.mean(axis=0)
    for k in topk:
        key = "{}@{}".format(metric, k)
        metric_dict[key] = round(avg_result[k - 1], decimal_place)
    return metric_dict


def get_metrics(indices, training_args):
    pos_index = np.array(indices == 0)
    pos_len = np.ones(len(pos_index), dtype=np.int8)
    topk = training_args.topk
    decimal_place = training_args.decimal_place
    metrics = {}
    metrics.update(topk_result("MRR", mrr_(pos_index, pos_len), topk, decimal_place))
    metrics.update(topk_result("NDCG", ndcg_(pos_index, pos_len), topk, decimal_place))
    metrics.update(topk_result("Recall", recall_(pos_index, pos_len), topk, decimal_place))

    return metrics
