import pytest
import numpy as np
from QBM import QBM_model


@pytest.fixture
def model():
    model = QBM_model(2, 1000)
    return model

def test_sigma(model):
    assert np.array_equal(model.sigma_x(0), [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
    assert np.array_equal(model.sigma_x(1), [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert np.array_equal(model.sigma_z(0), [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
    assert np.array_equal(model.sigma_z(1), [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

def test_lambda_v(model):
    N = model.N
    v_state = model.get_v_state()
    for i in range(2**N):
        m = model.iden(N)
        for v in range(N):
            m *= ((model.iden(N) + v_state[i][v] * model.sigma_z(v)))/2
        assert np.array_equal(model.lambda_v(i), m)
