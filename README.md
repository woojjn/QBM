# QBM

양자 볼츠만 머신
Reference:
Amin, M. H., Andriyash, E., Rolfe, J., Kulchytskyy, B. & Melko, R. (2018, May 23).

    Quantum Boltzmann Machine. Phys. Rev. Vol. 8. Retrieved from 
    
    https://arxiv.org/pdf/1601.02036.pdf

# QBM_model의 parameter

N : 큐빗의 개수

training_set :  훈련 셋의 개수

p, M : 예시 데이터[Pv_data_example]를 만들 때 정하는 매개변수 (Referecne 식[53])

seed : 예시 데이터[Pv_data_example]와 이를 몬테카를로 시뮬레이션 한 [Pv_data]를 고정시키기 위한 seed

-----------------------------------------------------------------------------------------

# model 생성 후 실행
model = QBM_model(N, training_set, p=0.9, M=8, seed=None)으로 인스턴스를 생성할 수 있습니다.

model.minimize_BM(x0)

model.minimize_BM(x0)

model.differential_evolution_BM(bounds)

model.differential_evolution_BM(bounds)

위의 함수들을 통해 BM[QBM]에 대해 minimize[differential_evolution]를 실행합니다.

model.result에 그 값이 저장됩니다. 이 값은 model.get_result()함수를 통해 얻을 수 있습니다.

# 결과값

result = model.get_result()

결과값은 다음과 같이 dict형태로 저장됩니다.

result = {'BM_KL' : [], 'QBM_KL' : [], 'BM_result' : None, 'QBM_result' : None}

