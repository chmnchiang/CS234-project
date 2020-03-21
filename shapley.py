import random
from utils import eprint
from tqdm import tqdm

def data_shapley(data, eval_metric, n_iter=30):

    n = len(data)
    indices = list(range(n))
    result = [0.0 for _ in range(n)]

    for it in range(n_iter):
        eprint(f'Shapley iter #{it}:')

        random.shuffle(indices)

        current_list = []
        v_last = eval_metric(current_list)

        for idx in tqdm(indices):
            current_list.append(data[idx])
            v_now = eval_metric(current_list)

            result[idx] += (v_now - v_last) / n_iter
            v_last = v_now

        temp_result = list(zip(result, range(n)))
        temp_result.sort()
        temp_result = [(f'{v:.2f}', idx) for v, idx in temp_result[::-1]]
        eprint(temp_result)

    return result

