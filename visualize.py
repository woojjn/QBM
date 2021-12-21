from QBM import pickle_data_processing

Ns = range(2, 7)
ts = [1000, 2000, 4000, 8000, 16000, 32000]

pickle_data_processing(5, ts, trials=10).plot()