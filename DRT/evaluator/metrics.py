import math


def recall(indices, topk):
    result = [0] * len(topk)
    for indice in indices:
        for pos, isture in enumerate(indice):
            if isture != 0:
                for k in range(len(topk)):
                    if pos < topk[k]:
                        result[k] += 1
                break
    return result


def mrr(indices, topk):
    result = [0] * len(topk)
    for indice in indices:
        for pos, isture in enumerate(indice):
            if isture != 0:
                for k in range(len(topk)):
                    if pos < topk[k]:
                        result[k] += 1 / (pos + 1)
                break
    return result


def ndcg(indices, topk):
    result = [0] * len(topk)
    dcg = [0] * len(topk)
    idcg = [0] * len(topk)
    for indice in indices:
        # 1 = related, 0 = unrelated
        cnt = 0
        for n, item in enumerate(indice):
            if item:
                cnt += 1
                for k in range(len(topk)):
                    if n < topk[k]:
                        dcg[k] += 1.0 / math.log(n + 2)
        for i in range(max(cnt, 1)):
            for k in range(len(topk)):
                if i < topk[k]:
                    idcg[k] += 1.0 / math.log(i + 2)
    for i in range(len(topk)):
        result[i] = dcg[i] / idcg[i]
    return result


def get_metrics(indices, topk):
    metrics = ['Recall@', 'MRR@', 'NDCG@']
    r = recall(indices, topk)
    m = mrr(indices, topk)
    n = ndcg(indices, topk)
    result = {}
    for metric, data in zip(metrics, [r, m, n]):
        for k, v in zip(topk, data):
            result[metric + str(k)] = v
    return result
