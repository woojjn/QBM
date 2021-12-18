from QBM_module import *

def test_sigma():
    assert np.array_equal(sigma_x(1, 0), [[0, 1], [1, 0]])
    assert np.array_equal(sigma_x(2, 0), [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
    assert np.array_equal(sigma_x(2, 1), [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert np.array_equal(sigma_z(1, 0), [[1,0], [0, -1]])
    assert np.array_equal(sigma_z(2, 0), [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
    assert np.array_equal(sigma_z(2, 1), [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

def test_Lambda_v():
    v_state = [1, -1]
    N = np.log2(len(v_state))
    m = np.identity(2**N)
    for v in range(2**N):
        m = np.kron(m, (np.identity(2**N) + v_state[v] * sigma_z(N, v)/2))
    assert np.array_equal(Lambda_v(N, v_state), m)