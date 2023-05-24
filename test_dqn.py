import numpy as np
import pytest
import torch

import dqn

N_ACTIONS = 6
STATE_SHAPE = (4, dqn.FRAME_SIZE, dqn.FRAME_SIZE)


@pytest.fixture(name="device")
def fixture_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize(
    "input_shape, expected_output_shape",
    [
        (STATE_SHAPE, (N_ACTIONS,)),
        ((32, *STATE_SHAPE), (32, N_ACTIONS)),
    ],
)
def test_dqn(input_shape, expected_output_shape):
    x = torch.ones(input_shape)
    dqn_ = dqn.DQN(n_actions=N_ACTIONS)
    assert dqn_(x).shape == expected_output_shape


@pytest.mark.parametrize("batch_size", [2, 5, 10])
def test_memory(batch_size, device):
    memory = dqn.Memory(max_size=5, state_shape=STATE_SHAPE, device=device)
    s = torch.ones(STATE_SHAPE)
    for i in range(1, 8):
        memory.add(s, torch.tensor(i), 0, s, False)
        state, *_ = memory.sample(batch_size)
        assert state.shape == (batch_size, *STATE_SHAPE)
    assert memory.actions.flatten().tolist() == [6, 7, 3, 4, 5]

    state, action, reward, next_state, terminal = memory.sample(batch_size)
    assert state.shape == (batch_size, *STATE_SHAPE)
    assert action.shape == (batch_size, 1)
    assert reward.shape == (batch_size, 1)
    assert next_state.shape == (batch_size, *STATE_SHAPE)
    assert terminal.shape == (batch_size, 1)


@pytest.mark.parametrize(
    "i, expected_epsilon",
    [
        (0, 1),
        (5e5, 0.505),
        (1e6, 0.01),
        (5e6, 0.01),
    ],
)
def test_epsilon_schedule(i, expected_epsilon):
    epsilon_start = 1
    epsilon_end = 0.01
    exploration_fraction = 0.1
    n_iter = int(1e7)
    epsilon_schedule = dqn.EpsilonSchedule(epsilon_start, epsilon_end, exploration_fraction, n_iter)
    assert epsilon_schedule(i) == pytest.approx(expected_epsilon)


def test_preprocess(device):
    frame = np.random.randint(256, size=(210, 160, 3), dtype=np.uint8)
    x = dqn.preprocess(frame, device).cpu()
    assert x.shape == (1, dqn.FRAME_SIZE, dqn.FRAME_SIZE)
    assert x.min() == pytest.approx(0)
    assert x.max() == pytest.approx(1)
