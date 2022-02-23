import os
import copy
import pickle
import random
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import hamming


class QBM_model:
    def __init__(self, N, training_set, p=0.9, M=8, seed=None):
        """
        Reference:
        Amin, M. H., Andriyash, E., Rolfe, J., Kulchytskyy, B. & Melko, R. (2018, May 23).
            Quantum Boltzmann Machine. Phys. Rev. Vol. 8. Retrieved from 
            https://arxiv.org/pdf/1601.02036.pdf


        N : 큐빗의 개수
        training_set :  훈련 셋의 개수
        p, M : 예시 데이터[Pv_data_example]를 만들 때 정하는 매개변수 (Referecne 식[53])
        seed : 예시 데이터[Pv_data_example]와 이를 몬테카를로 시뮬레이션 한 [Pv_data]를 고정시키기 위한 seed
        """
        
        self.N = N
        self.M = M
        self.training_set = training_set

        # save값의 default는 True입니다. save값을 True로 하면 minimize를 실행하는 동안의 KL이 저장됩니다.
        self.save = True
        
        # 결과는 6가지의 key를 가진 dict형태로 저장됩니다. 'BM_KL'과 'QBM_KL'은 minimize하는 동안 KL이 저장되는 key입니다.
        # 'Pv_data_example'과 'Pv_data'는 예시 데이터와 훈련 데이터를 저장합니다.
        # 'BM_result', 'QBM_result'은 minimize한 결과를 보여줍니다.
        self.result = {'Pv_data_example' : None, 'Pv_data' : None, 'BM_KL' : [], 'QBM_KL' : [], 'BM_result' : None, 'QBM_result' : None}

        # random seed 설정
        if seed:
            random.seed(seed)


        # visible state 생성
        self.v_state = []

        for i in range(2**N):
            vi = bin(2**N - i - 1).split('b')[1].replace("1", "1 ").replace("0", "-1 ").split()
            vi.reverse()
            vi += [-1]*(N - len(vi))
            vi.reverse()
            self.v_state.append(vi)

        self.v_state = np.array(self.v_state, dtype=int)


        # s_state 무작위적으로 생성
        s_state = np.zeros([M, N], dtype=int)

        for k in range(M):
            for i in range(N):
                r = random.choice([-1, 1])
                # r = random.choices([-1, 1], weights=[1 - p, p])[0]
                s_state[k][i] = r
        
        
        # v_state와 s_state[k]의 해밍 거리: dv[k][v_state]
        dv = np.zeros([M, 2**N], dtype=int)

        for k in range(M):
            for v in range(2**N):
                dv[k][v] = hamming(s_state[k], self.v_state[v]) * N


        # 예시 데이터 Pv_data_example[v] 생성
        self.Pv_data_example = np.zeros(2**N)

        for v in range(2**N):
            for k in range(M):
                self.Pv_data_example[v] += p**(N - dv[k][v])*(1 - p)**dv[k][v] / M

        self.result['Pv_data_example'] =  self.Pv_data_example

        # training_set만큼 예시 데이터에 대한 몬테카를로 시뮬레이션을 실행한 후의 확률 분포 Pv_data[v]
        self.Pv_data= np.zeros(2**N)

        prob_line = np.cumsum(self.Pv_data_example)
        for _ in range(training_set):
            r = random.random()
            for i in range(2**N):
                if i == 0 and r<=prob_line[0]:
                    self.Pv_data[0] += 1
                elif prob_line[i - 1]<r<=prob_line[i]:
                    self.Pv_data[i] += 1
                
        self.Pv_data /= training_set

        self.result['Pv_data'] =  self.Pv_data


    def iden(self, N):
        # 2**N차원 identity matrix

        M = np.identity(2**N)
        return M

    def sigma_x(self, i):
        # 2**N 차원 행렬
        # 0<=i<N

        s_x = np.array([[0, 1], [1, 0]])    # pauli matrix(x)
        if not 0<=i<self.N:
            raise IndexError("i는 0보다 크거나 같고 N보다는 작아야 합니다.")

        M = self.iden(i)
        M = np.kron(M, s_x)
        M = np.kron(M, self.iden(self.N - i - 1))
        
        return M

    def sigma_z(self, i):
        # 2**N 차원 행렬
        # 0<=i<N

        s_z = np.array([[1, 0], [0, -1]])    # pauli matrix(z)
        if not 0<=i<self.N:
            raise IndexError("i는 0보다 크거나 같고 N보다는 작아야 합니다.")

        M = self.iden(i)
        M = np.kron(M, s_z)
        M = np.kron(M, self.iden(self.N - i - 1))
        
        return M

    def lambda_v(self, v):
        # 2**N 차원 행렬
        # v번째 대각 성분이 1이고, 나머지가 0인 대각 행렬
        # 0<=i<2**N

        if not 0<=v<2**self.N:
            raise IndexError("i는 0보다 크거나 같고 2**N보다는 작아야 합니다.")

        m = [0] * v
        m += [1]
        m += [0] * (2**self.N - v - 1)
        M = np.diag(m)
        return M

    def BM(self, theta):
        # 해밀토니언
        H = np.zeros([2**self.N, 2**self.N])

        # 매개변수
        b = np.zeros(self.N)
        w = np.zeros([self.N, self.N])


        # 매개변수 행렬 생성
        for i in range(self.N):
            b[i] = theta[i]
            for j in range(self.N):
                if i < j:
                    w[i][j] = theta[self.N - 1 + int((self.N - 3/2 - i/2) * i + j)]
                    w[j][i] = w[i][j]


        # 해밀토니언 계산
        for i in range(self.N):
            H -= b[i] * self.sigma_z(i)
            for j in range(self.N):
                H -= w[i][j] * np.dot(self.sigma_z(i), self.sigma_z(j))

        Z = np.trace(expm(-H))
        rho = expm(-H)/Z

        KL = 0

        for v in range(2**self.N):
            Pv = np.trace(np.dot(self.lambda_v(v), rho))
            L = -self.Pv_data[v] * np.log(Pv)
            if self.Pv_data[v] == 0:
                L_min = 0
            else:
                L_min = -self.Pv_data[v] * np.log(self.Pv_data[v])

            KL += L - L_min
        
        if self.save:
            self.result['BM_KL'].append(KL)

        return KL

    def QBM(self, theta):
        # 해밀토니언
        H = np.zeros([2**self.N, 2**self.N])

        # 매개변수
        gamma = theta[0]
        b = np.zeros(self.N)
        w = np.zeros([self.N, self.N])


        # 매개변수 행렬 생성
        for i in range(self.N):
            b[i] = theta[i + 1]
            for j in range(self.N):
                if i < j:
                    w[i][j] = theta[self.N + int((self.N - 3/2 - i/2) * i + j)]
                    w[j][i] = w[i][j]


        # 해밀토니언 계산
        for i in range(self.N):
            H -= gamma * self.sigma_x(i) + b[i] * self.sigma_z(i)
            for j in range(self.N):
                H -= w[i][j] * np.dot(self.sigma_z(i), self.sigma_z(j))

        Z = np.trace(expm(-H))
        rho = expm(-H)/Z

        KL = 0

        for v in range(2**self.N):
            Pv = np.trace(np.dot(self.lambda_v(v), rho))
            L = -self.Pv_data[v] * np.log(Pv)
            if self.Pv_data[v] == 0:
                L_min = 0
            else:
                L_min = -self.Pv_data[v] * np.log(self.Pv_data[v])

            KL += L - L_min
        
        if self.save:
            self.result['QBM_KL'].append(KL)

        return KL

    # scipy.optimize.minimize
    def minimize_BM(self, x0=None, save=True, args=(), method="BFGS", jac=None,
                     hess=None, hessp=None, bounds=None, constraints=(), 
                     tol=None, callback=None, options=None):

        """
        x0를 정하지 않으면, 모든 값이 0.1로 고정됩니다.
        save를 False로 정하면, minimize 동안 KL 값이 저장되지 않습니다.
        method의 기본값은 "BFGS"입니다.
        """

        if not save:
            self.save = False
        else:
            self.save = True
        
        if x0 == None:
            x0 = [0.1] * ((self.N**2 + self.N)//2)
        elif len(x0) != (self.N**2 + self.N)//2:
            raise Exception("매개변수의 개수[(N**2 + N)/2]만큼 x0를 설정해야 합니다.")

        result = minimize(self.BM, x0, args=args, method=method, jac=jac, hess=hess,
                          hessp=hessp, bounds=bounds, constraints=constraints,
                          tol=tol, callback=callback, options=options)

        if not result.success:
            print("수렴에 실패했습니다.")

        self.result["BM_result"] = result
        
        return result

    # scipy.optimize.minimize
    def minimize_QBM(self, x0=None, save=True, args=(), method="BFGS", jac=None,
                     hess=None, hessp=None, bounds=None, constraints=(), 
                     tol=None, callback=None, options=None):

        """
        x0를 정하지 않으면, 모든 값이 0.1로 고정됩니다.
        save를 False로 정하면, minimize 동안 KL 값이 저장되지 않습니다.
        method의 기본값은 "BFGS"입니다.
        """

        if not save:
            self.save = False
        else:
            self.save = True
        
        if x0 == None:
            x0 = [0.1] * ((self.N**2 + self.N)//2 + 1)

        elif len(x0) != (self.N**2 + self.N)//2 + 1:
            raise Exception("매개변수의 개수[(N**2 + N)/2 + 1]만큼 x0를 설정해야 합니다.")

        result = minimize(self.QBM, x0=x0, args=args, method=method, jac=jac, hess=hess,
                          hessp=hessp, bounds=bounds, constraints=constraints,
                          tol=tol, callback=callback, options=options)

        if not result.success:
            print("수렴에 실패했습니다.", self.Pv_data_example, self.Pv_data, sep='\n')

        self.result["QBM_result"] = result
        
        return result

    # scipy.optimize.differentail_evolution
    def differential_evolution_BM(self, save=True, bounds=None, args=(), strategy='best1bin',
                                  maxiter=1000, popsize=15, tol=0.01,
                                  mutation=(0.5, 1), recombination=0.7, seed=None,
                                  callback=None, disp=False, polish=True,
                                  init='latinhypercube', atol=0, updating='immediate',
                                  workers=1, constraints=()):

        """
        bounds를 정하지 않으면, 모든 값이 (-10, 10)으로 고정됩니다.
        save를 False로 정하면, differential_evolution 동안 KL 값이 저장되지 않습니다.
        """

        if not save:
            self.save = False
        else:
            self.save = True
        
        if bounds == None:
            bounds = [(-10, 10)] * ((self.N**2 + self.N)//2)
        elif len(bounds) != (self.N**2 + self.N)//2:
            raise Exception("매개변수의 개수[(N**2 + N)/2]만큼 bounds를 설정해야 합니다.")

        result = differential_evolution(self.BM, bounds=bounds, args=args, strategy=strategy,
                                        maxiter=maxiter, popsize=popsize, tol=tol,
                                        mutation=mutation, recombination=recombination, seed=seed,
                                        callback=callback, disp=disp, polish=polish,
                                        init=init, atol=atol, updating=updating,
                                        workers=workers, constraints=constraints)

        self.result["BM_result"] = result
        
        return result

    # scipy.optimize.differential_evolution
    def differential_evolution_QBM(self, save=True, bounds=None, args=(), strategy='best1bin',
                                  maxiter=1000, popsize=15, tol=0.01,
                                  mutation=(0.5, 1), recombination=0.7, seed=None,
                                  callback=None, disp=False, polish=True,
                                  init='latinhypercube', atol=0, updating='immediate',
                                  workers=1, constraints=()):

        """
        bounds를 정하지 않으면, 모든 값이 (-10, 10)으로 고정됩니다.
        save를 False로 정하면, differential_evolution 동안 KL 값이 저장되지 않습니다.
        """

        if not save:
            self.save = False
        else:
            self.save = True
        
        if bounds == None:
            bounds = [(-10, 10)] * ((self.N**2 + self.N)//2 + 1)

        elif len(bounds) != (self.N**2 + self.N)//2 + 1:
            raise Exception("매개변수의 개수[(N**2 + N)/2 + 1]만큼 bounds를 설정해야 합니다.")

        result = differential_evolution(self.QBM, bounds=bounds, args=args, strategy=strategy,
                                        maxiter=maxiter, popsize=popsize, tol=tol,
                                        mutation=mutation, recombination=recombination, seed=seed,
                                        callback=callback, disp=disp, polish=polish,
                                        init=init, atol=atol, updating=updating,
                                        workers=workers, constraints=constraints)

        self.result["QBM_result"] = result
        
        return result

    # 결과값 pickle로 저장하기
    def result_to_pickle(self, dir_name = None, file_name=None):
        import pickle
        result = self.get_result()

        # result 디렉토리가 없으면 생성
        if not os.path.exists("result"):
            os.mkdir("result")

        if dir_name == None:
            dir_name = f"N{self.N}"

        path = f"result/{dir_name}/"

        # result/dir_name 디렉토리가 없으면 생성
        if not os.path.exists(path):
            os.mkdir(path)

        # 덮어쓰기 방지
        if file_name == None:
            trial = 0
            file_name = f"t{self.training_set}_M{self.M}_trial{trial}.pickle"

            while os.path.exists(path + file_name):
                trial += 1
                file_name = f"t{self.training_set}_M{self.M}_trial{trial}.pickle"

        file_name = path + file_name

        # pickle파일 만들기
        with open(file_name, 'wb') as f:
            pickle.dump(result, f)

        return print(file_name, "저장되었습니다.")

    def plot_from_result(self, options=["BM", "QBM"]):
        import matplotlib.pyplot as plt
        result = self.get_result()
            
        for option in options:
            print(f"{option}\t결과값", end=' : ')
            option += "_result"
            print(result[option].fun)

        for option in options:
            option += "_KL"
            plt.plot(range(len(result[option])), result[option])

        plt.xlabel("iteration")
        plt.ylabel("KL")
        plt.legend(options)

        plt.show()


    def get_result(self):
        return self.result

    def get_Pv_data(self):
        return self.Pv_data

    def get_Pv_data_example(self):
        return self.Pv_data_example

    def get_v_state(self):
        return self.v_state

# pickle 데이터 처리
class PickleDataProcessing:
    def __init__(self, Ns, training_sets, M=8, trial_range=(0, 10), path="result/"):
        """
        path + f"N{Ns}/t{training_sets}_M{M}+trial{trial}.pickle" 데이터를 얻습니다.
        

        Ns : list
        training_sets : int

        or

        Ns : int
        training_sets : list

        와 같이 변수를 정해주세요.

        만약 Ns를 하나의 값으로(int) 정하면, get_N_data 함수를 통해 training_sets의 데이터들을 얻을 수 있습니다.
        training_sets의 경우는 get_training_set_data 함수를 통해 얻을 수 있습니다.

        둘 다 int 타입으로 정할 경우, get_single_data 함수를 통해 값을 얻을 수 있습니다.
        """
        self.Ns = Ns
        self.training_sets = training_sets
        self.datatype = None

        if type(trial_range) == int:
            self.trial = trial_range
        elif type(trial_range) == tuple:
            self.start, self.end = trial_range
            self.range = range(self.start, self.end)

        # single_data(pickle 파일 하나만 분석)
        if type(Ns) == int and type(training_sets) == int and type(trial_range) == int:
            file_name = path + f"N{Ns}/t{training_sets}_M{M}_trial{self.trial}.pickle"
            with open(file_name, "rb") as f:
                data = pickle.load(f)

            self.datatype = "single"
            self.single_data = data

        # N_data(training_set은 고정, N에 따른 데이터 분석)
        elif type(training_sets) == int:
            self.N_data = {"BM":[[] for _ in self.range], "QBM":[[] for _ in self.range]}
            for i, trial in enumerate(self.range):
                for N in Ns:
                    file_name = path + f"N{N}/t{training_sets}_M{M}_trial{trial}.pickle"
                    with open(file_name, "rb") as f:
                        data = pickle.load(f)

                    self.N_data["BM"][i].append(data["BM_result"].fun)
                    self.N_data["QBM"][i].append(data["QBM_result"].fun)

            self.datatype = "N"
            self.N_data["BM"] = np.array(self.N_data["BM"])
            self.N_data["QBM"] = np.array(self.N_data["QBM"])
            
        # training_set_data(N은 고정, training_set에 따른 데이터 분석)
        elif type(Ns) == int:
            self.training_set_data = {"BM":[[] for _ in self.range], "QBM":[[] for _ in self.range]}
            for i, trial in enumerate(self.range):
                for training_set in training_sets:
                    file_name = path + f"N{Ns}/t{training_set}_M{M}_trial{trial}.pickle"
                    with open(file_name, "rb") as f:
                        data = pickle.load(f)

                    self.training_set_data["BM"][i].append(data["BM_result"].fun)
                    self.training_set_data["QBM"][i].append(data["QBM_result"].fun)

            self.datatype = "training_set"
            self.training_set_data["BM"] = np.array(self.training_set_data["BM"])
            self.training_set_data["QBM"] = np.array(self.training_set_data["QBM"])

        else:
            raise TypeError("Ns 혹은 training_sets 중 하나를 list로 정해주세요.")

    
    # plot
    """
    single_data : iteration에 따른 KL값
    N_data : N에 따른 KL값
    training_set_data : training_set에 따른 KL값
    """
    def plot(self, xscale="linear", ylim=None):
        import matplotlib.pyplot as plt

        """
        xscale : xscale을 'log'로 설정 가능
        ylim : y의 최대 최솟값 설정 가능 tuple
        top : trial 중 가장 작은 값 top개만 선정하여 plot
        """

        if self.datatype == "single":
            single_data = self.get_single_data()

            plt.plot(range(len(single_data["BM_KL"])), single_data["BM_KL"])
            plt.plot(range(len(single_data["QBM_KL"])), single_data["QBM_KL"])
            
            plt.title(f"N={self.Ns}, training set={self.training_sets}, trial={self.trials}")
            plt.xlabel("iteration")
            plt.ylabel("KL")
            plt.ylim(ylim)
            plt.legend(["BM", "QBM"])


        elif self.datatype == "N":
            Ns = self.Ns
            N_data = self.get_N_data()

            BM_yerr = [N_data["BM_std"], N_data["BM_std"]]
            QBM_yerr = [N_data["QBM_std"], N_data["QBM_std"]]
            plt.errorbar(Ns, N_data["BM"], yerr=BM_yerr, fmt='o', capsize=5, c="royalblue")
            plt.errorbar(Ns, N_data["QBM"], yerr=QBM_yerr, fmt='o', capsize=5, c="tomato")

            plt.title(f"training set={self.training_sets}")
            plt.xticks(self.Ns, map(str, map(int, self.Ns)))
            plt.xlabel("N")
            plt.ylabel("KL")
            plt.ylim(ylim)
            plt.legend(["BM", "QBM"])

        elif self.datatype == "training_set":
            training_sets = self.training_sets
            training_set_data = self.get_training_set_data()

            BM_yerr = [training_set_data["BM_std"], training_set_data["BM_std"]]
            QBM_yerr = [training_set_data["QBM_std"], training_set_data["QBM_std"]]
            plt.errorbar(training_sets, training_set_data["BM"], yerr=BM_yerr, fmt='o', capsize=5, c="royalblue")
            plt.errorbar(training_sets, training_set_data["QBM"], yerr=QBM_yerr, fmt='o', capsize=5, c="tomato")

            plt.title(f"N={self.Ns}")
            plt.xlabel("training_set")
            plt.ylabel("KL")
            plt.ylim(ylim)
            plt.legend(["BM", "QBM"])
            plt.xscale(xscale)

        plt.show()

    def get_single_data(self):
        return self.single_data

    def get_N_data(self, mean=True):
        data = copy.deepcopy(self.N_data)
        if mean:
            for i in range(len(self.Ns)):
                data["BM"] = np.mean(self.N_data["BM"], axis=0)
                data["QBM"] = np.mean(self.N_data["QBM"], axis=0)
                data["BM_std"] = np.std(self.N_data["BM"], axis=0)
                data["QBM_std"] = np.std(self.N_data["QBM"], axis=0)
            return data
        else:
            return data

    def get_training_set_data(self, mean=True):
        data = copy.deepcopy(self.training_set_data)
        if mean:
            for i in range(len(self.training_sets)):
                data["BM"] = np.mean(self.training_set_data["BM"], axis=0)
                data["QBM"] = np.mean(self.training_set_data["QBM"], axis=0)
                data["BM_std"] = np.std(self.training_set_data["BM"], axis=0)
                data["QBM_std"] = np.std(self.training_set_data["QBM"], axis=0)
            return data
        else:
            return data