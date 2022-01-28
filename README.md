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


# model 생성 후 실행
model = QBM_model(N, training_set, p=0.9, M=8, seed=None)로 인스턴스를 생성할 수 있습니다.

    model.minimize_BM(x0)

    model.minimize_BM(x0)

    model.differential_evolution_BM(bounds)

    model.differential_evolution_BM(bounds)

위의 함수들을 통해 BM[QBM]에 대해 minimize[differential_evolution]를 실행합니다.

model.result에 그 값이 저장됩니다. 이 값은 model.get_result()함수를 통해 얻을 수 있습니다.

# 결과값

    result = model.get_result()

결과값은 다음과 같이 dict형태로 저장됩니다.

    result = {'Pv_data_example' : None, 'Pv_data' : None, 'BM_KL' : [], 'QBM_KL' : [], BM_result' : None, 'QBM_result' : None}

-----------------------------------------------------------------------------------------

# PickleDataProcessing

저장된 pickle 데이터를 분석합니다.

data = PickleDataProcessing(Ns, training_sets, M=8, trials=10, path='/result')으로 인스턴스를 생성할 수 있습니다.

path + f"N{Ns}/t{training_sets}_M{M}+trial{trials}.pickle" 에 있는 데이터를 얻습니다.

trials 값은 같은 실험에 대해 여러 개의 trial을 얻을 때 쓰입니다.

trials=t 라고 설정한 경우, trials는 range(t)의 trail 데이터들을 얻습니다.

# single_data
    Ns : int

    training_sets : int
    
# N_data
        
    Ns : list

    training_sets : int

# training_set_dat

    Ns : int
    
    training_sets : list

변수의 타입에 따라 위의 데이터 타입을 얻을 수 있습니다.

각 데이터 타입에 따라 그 결과 값을 다음과 같이 얻을 수 있습니다.

    data.get_single_data()
    
    data.get_N_data(mean=True, top=None)
    
    data.get_training_set_data(mean=True, top=None)
    
mean은 여러 개의 trial 데이터 값을 평균낼 지를 결정합니다.

top은 여러 개의 시도 중 가장 작은 값을 얻은 KL 값 top개를 평균냅니다.

# plot

PickleDataProcessing은 plot 함수가 있습니다. 

    data.plot(xscale='linear', ylim=None, top=None)
    
single_data의 경우, iteration에 따른 KL값을 plot합니다.

N_data의 경우, N에 따른 KL의 평균값을 plot합니다.

training_set_data의 경우, training_set에 따른 KL의 평균값을 plot합니다.

top을 설정한 경우, trial 중 가장 작은 top개의 trial 데이터를 평균 내어 plot합니다.
