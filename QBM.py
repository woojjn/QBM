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

        # save값의 default는 True입니다. save값을 True로 하면 minimize를 실행하는 동안의 KL이 저장됩니다.
        self.save = True
        
        # 결과는 4가지의 key를 가진 dict형태로 저장됩니다. 'BM_KL'과 'QBM_KL'은 minimize하는 동안 KL을 저장시켜 보여줍니다.
        # 'BM_result', 'QBM_result'은 minimize한 결과를 보여줍니다. 이는 save값에 상관없이 저장됩니다.
        self.result = {'BM_KL' : [], 'QBM_KL' : [], 'BM_result' : None, 'QBM_result' : None}

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
            try:
                L_min = -self.Pv_data[v] * np.log(self.Pv_data[v])
            except:
                L_min = 0

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
            try:
                L_min = -self.Pv_data[v] * np.log(self.Pv_data[v])
            except:
                L_min = 0

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

        result = minimize(self.QBM, x0, args=args, method=method, jac=jac, hess=hess,
                          hessp=hessp, bounds=bounds, constraints=constraints,
                          tol=tol, callback=callback, options=options)

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

    def get_result(self):
        return self.result

    def get_Pv_data(self):
        return self.Pv_data

    def get_Pv_data_example(self):
        return self.Pv_data_example

    def get_v_state(self):
        return self.v_state


model = QBM_model(N=4, training_set=1000)
model.minimize_BM()
model.minimize_QBM()

result = model.get_result()

print(result["BM_KL"][:5])
print(result["QBM_KL"][:5])
print(result["BM_result"])
print(result["QBM_result"])