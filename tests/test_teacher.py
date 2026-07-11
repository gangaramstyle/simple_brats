import inspect

import torch

from simple_brats.models import BlindPatchTeacher, EMATeacher


def test_teacher_api_is_patch_only_and_permutation_equivariant() -> None:
    torch.manual_seed(10)
    teacher = BlindPatchTeacher(embedding_dim=16, hidden_dim=12, patch_shape=(4, 4, 1)).eval()
    patches = torch.randn(2, 5, 4, 4, 1)
    permutation = torch.tensor([3, 0, 4, 1, 2])

    parameters = list(inspect.signature(BlindPatchTeacher.forward).parameters)
    assert parameters == ["self", "patches"]
    expected = teacher(patches)
    actual = teacher(patches[:, permutation])

    torch.testing.assert_close(actual, expected[:, permutation])


def test_teacher_encodes_patches_independently() -> None:
    torch.manual_seed(11)
    teacher = BlindPatchTeacher(embedding_dim=8, patch_shape=(4, 4, 1)).eval()
    patches = torch.randn(1, 3, 4, 4, 1)
    changed = patches.clone()
    changed[:, 1] += 100

    original_output = teacher(patches)
    changed_output = teacher(changed)

    torch.testing.assert_close(original_output[:, 0], changed_output[:, 0])
    torch.testing.assert_close(original_output[:, 2], changed_output[:, 2])
    assert not torch.allclose(original_output[:, 1], changed_output[:, 1])


def test_ema_teacher_updates_without_gradients() -> None:
    torch.manual_seed(12)
    online = BlindPatchTeacher(embedding_dim=8, patch_shape=(4, 4, 1))
    ema = EMATeacher(online, momentum=0.5)
    ema.train()
    assert not ema.teacher.training
    old_parameters = [parameter.clone() for parameter in ema.teacher.parameters()]
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.add_(1.0)
    ema.update(online)
    assert ema.num_updates.item() == 1

    for old, target, source in zip(
        old_parameters, ema.teacher.parameters(), online.parameters(), strict=True
    ):
        torch.testing.assert_close(target, old.lerp(source, 0.5))
        assert not target.requires_grad
