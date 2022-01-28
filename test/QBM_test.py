import pytest
import pickle
import numpy as np
from scipy.linalg import expm
from QBM import QBM_model, PickleDataProcessing


@pytest.fixture
def model():
    model = QBM_model(4, 1000)
    return model

@pytest.fixture
def pickle_data():
    return PickleDataProcessing(4, 1000)
     

def test_pickle(model, pickle_data):
    result = pickle_data.get_single_data()
    theta = result["BM_result"].x
    Pv_data = result["Pv_data"]
    N = pickle_data.Ns
    H = np.zeros([2**N, 2**N])

    # 매개변수
    b = np.zeros(N)
    w = np.zeros([N, N])


    # 매개변수 행렬 생성
    for i in range(N):
        b[i] = theta[i]
        for j in range(N):
            if i < j:
                w[i][j] = theta[N - 1 + int((N - 3/2 - i/2) * i + j)]
                w[j][i] = w[i][j]


    # 해밀토니언 계산
    for i in range(N):
        H -= b[i] * model.sigma_z(i)
        for j in range(N):
            H -= w[i][j] * np.dot(model.sigma_z(i), model.sigma_z(j))

    Z = np.trace(expm(-H))
    rho = expm(-H)/Z
    P = []

    for v in range(2**N):
        Pv = np.trace(np.dot(model.lambda_v(v), rho))
        P.append(Pv)

    assert np.array_equal(P, Pv_data)

# def test_sigma(model):
#     assert np.array_equal(model.sigma_x(0), [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
#     assert np.array_equal(model.sigma_x(1), [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
#     assert np.array_equal(model.sigma_z(0), [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
#     assert np.array_equal(model.sigma_z(1), [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

# def test_lambda_v(model):
#     N = model.N
#     v_state = model.get_v_state()
#     for i in range(2**N):
#         print(i)
#         m = model.iden(N)
#         for v in range(N):
#             print(m)
#             m *= (model.iden(N) + v_state[i][v] * model.sigma_z(v))/2
#             print(m)
#             print()
#         assert np.array_equal(model.lambda_v(i), m)

# def test_v_state(model):
#     v_state = model.get_v_state()
#     assert np.array_equal(v_state[0], [1, 1])
#     assert np.array_equal(v_state[1], [1, -1])
#     assert np.array_equal(v_state[2], [-1, 1])
#     assert np.array_equal(v_state[3], [-1, -1])

