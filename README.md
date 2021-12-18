# BM # QBM # python

양자 볼츠만 머신
Reference: Quantum Boltzmann Machine, Retrieved from https://arxiv.org/pdf/1601.02036.pdf

-----------------------------------------------------------------------------------------

model = model(N, training_set, p=0.9, M=8, seed=None)으로 인스턴스를 생성할 수 있습니다.

p, M은 Reference의 (53)번 식의 매개변수로서, p는 확률 매개변수, M은 모드의 개수를 의미합니다.

-----------------------------------------------------------------------------------------

model.minimize_BM(x0=None, save=True), model.minimize_QBM(x0=None, save=True) 을 통해 각각 BM과 QBM에 대해 minimize(method='BFGS')를 실행합니다.

x0는 parameter(gamma, w, b)의 초기값으로, None이면 모두 0.1의 값을 가집니다.

save=True이면 KL값을 model.result에 저장합니다. 그 결과는 model.get_result()를 통해 얻을 수 있습니다.
