from tqdm import tqdm
from QBM import QBM_model

Ns = range(2, 7)
ts = [1000, 2000, 4000, 8000, 16000, 32000]

for trial in tqdm(range(10)):
    for N in Ns:
        for training_set in ts:
            model = QBM_model(N, training_set)
            model.minimize_BM()
            model.minimize_QBM()
            model.result_to_pickle()
