from QBM import PickleDataProcessing

Ns = range(2, 7)
ts = [1000, 2000, 4000, 8000, 16000, 32000]
trial=20    
result = PickleDataProcessing(4, ts, trials=(20, 30)).plot()


# for i in range(trial):
    