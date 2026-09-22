import numpy as np
import torch

from processes.model import ExerciseModel


def test_prepare_movement_input_uses_uniform_fixed_shape():
    frames = [np.zeros((80, 120, 3), dtype=np.uint8) for _ in range(7)]

    model = ExerciseModel()
    movement = model.prepare_movement_input(frames)

    assert movement.shape == (1, 3, model.SEQUENCE_LENGTH, model.FRAME_SIZE, model.FRAME_SIZE)
    assert movement.dtype == torch.float32


def test_model_forward_supports_prepared_input_shape():
    model = ExerciseModel().eval()
    inputs = torch.zeros(1, 3, model.SEQUENCE_LENGTH, model.FRAME_SIZE, model.FRAME_SIZE)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs['phase'].shape == (1, 3)
    assert outputs['form'].shape == (1, 2)
    assert outputs['cycle'].shape == (1, 2)
