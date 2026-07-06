"""Integration tests for training loop components."""
import pytest
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from model1_research.model import MultiModel, WeightedSumLoss
from model1_research.train import train_one_epoch, validate


def _make_loaders():
    """Create tiny DataLoaders with batch_size = dataset_size (1 batch per epoch).

    SmileModel.forward sets requires_grad=True on input tensors, so
    multi-batch iteration over CPU slices of the same storage causes
    'backward through graph a second time' errors. Using a single
    batch per loader avoids this issue entirely.
    """
    torch.manual_seed(42)
    N = 16
    tau = torch.rand(N, 1) * 1.98 + 0.02
    logm = torch.rand(N, 1) * 1.0 - 0.5
    y = torch.rand(N, 1) * 0.099 + 0.001
    yATM = torch.rand(N, 1) * 0.099 + 0.001
    train_ds = TensorDataset(tau, logm, y, yATM)
    train_loader = DataLoader(train_ds, batch_size=N, shuffle=False)

    val_ds = TensorDataset(tau.clone(), logm.clone(), y.clone(), yATM.clone())
    val_loader = DataLoader(val_ds, batch_size=N, shuffle=False)

    N_c6 = 12
    tau_c6 = torch.rand(N_c6, 1) * 1.98 + 0.02
    logm_c6 = torch.cat([
        torch.rand(N_c6 // 2, 1) * (-2.0) - 1.0,
        torch.rand(N_c6 - N_c6 // 2, 1) * 2.0 + 1.0,
    ])
    yATM_c6 = torch.rand(N_c6, 1) * 0.099 + 0.001
    c6_ds = TensorDataset(tau_c6, logm_c6, yATM_c6)
    c6_loader = DataLoader(c6_ds, batch_size=N_c6, shuffle=False)

    return train_loader, val_loader, c6_loader


@pytest.fixture
def training_setup():
    """Fresh model, loss, optimizer, and single-batch loaders per test."""
    train_loader, val_loader, c6_loader = _make_loaders()
    model = MultiModel(hidden_sizes=[5, 5, 5], ensemble_num=2)
    loss_fn = WeightedSumLoss(weights=[1, 1, 10, 10, 10, 10])
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    device = torch.device('cpu')
    return {
        'model': model,
        'loss_fn': loss_fn,
        'optimizer': optimizer,
        'device': device,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'c6_loader': c6_loader,
    }


# ── train_one_epoch ───────────────────────────────────────────────────

class TestTrainOneEpoch:
    def test_returns_finite_loss(self, training_setup):
        s = training_setup
        loss = train_one_epoch(
            s['model'], s['train_loader'], s['c6_loader'],
            s['loss_fn'], s['optimizer'], s['device'],
        )
        assert isinstance(loss, float)
        assert not (loss != loss)  # not NaN

    def test_parameters_change(self, training_setup):
        s = training_setup
        params_before = {n: p.clone() for n, p in s['model'].named_parameters()}
        train_one_epoch(
            s['model'], s['train_loader'], s['c6_loader'],
            s['loss_fn'], s['optimizer'], s['device'],
        )
        changed = False
        for n, p in s['model'].named_parameters():
            if not torch.equal(params_before[n], p):
                changed = True
                break
        assert changed, "No parameters changed after training epoch"

    def test_gradient_clipping(self, training_setup):
        s = training_setup
        train_one_epoch(
            s['model'], s['train_loader'], s['c6_loader'],
            s['loss_fn'], s['optimizer'], s['device'],
            gradient_clip=0.1,
        )
        assert True


# ── validate ──────────────────────────────────────────────────────────

class TestValidate:
    def test_returns_finite_loss(self, training_setup):
        s = training_setup
        loss = validate(
            s['model'], s['val_loader'], s['c6_loader'],
            s['loss_fn'], s['device'],
        )
        assert isinstance(loss, float)
        assert not (loss != loss)

    def test_parameters_unchanged(self, training_setup):
        s = training_setup
        params_before = {n: p.clone() for n, p in s['model'].named_parameters()}
        validate(
            s['model'], s['val_loader'], s['c6_loader'],
            s['loss_fn'], s['device'],
        )
        for n, p in s['model'].named_parameters():
            assert torch.equal(params_before[n], p), f"Parameter {n} changed during validation"

    def test_autograd_works_in_eval(self, training_setup):
        """SmileModel needs autograd even in eval mode."""
        s = training_setup
        s['model'].eval()
        loss = validate(
            s['model'], s['val_loader'], s['c6_loader'],
            s['loss_fn'], s['device'],
        )
        assert isinstance(loss, float)


# ── Full training loop ────────────────────────────────────────────────

class TestFullLoop:
    def test_two_epochs_no_crash(self, training_setup):
        s = training_setup
        for epoch in range(2):
            train_loss = train_one_epoch(
                s['model'], s['train_loader'], s['c6_loader'],
                s['loss_fn'], s['optimizer'], s['device'],
            )
            val_loss = validate(
                s['model'], s['val_loader'], s['c6_loader'],
                s['loss_fn'], s['device'],
            )
        assert isinstance(train_loss, float)
        assert isinstance(val_loss, float)

    def test_scheduler_per_epoch_not_per_batch(self, training_setup):
        """Bug T4: scheduler.step() should be in epoch loop, not batch loop.

        MultiStepLR increments last_epoch on each step() call.
        Milestone [2] means LR drops when last_epoch reaches 2 (after 2 step() calls).
        """
        s = training_setup
        scheduler = optim.lr_scheduler.MultiStepLR(
            s['optimizer'], milestones=[2], gamma=0.5
        )
        lr_before = s['optimizer'].param_groups[0]['lr']

        # Epoch 0
        train_one_epoch(
            s['model'], s['train_loader'], s['c6_loader'],
            s['loss_fn'], s['optimizer'], s['device'],
        )
        scheduler.step()  # last_epoch -> 1
        lr_after_epoch0 = s['optimizer'].param_groups[0]['lr']
        assert lr_after_epoch0 == lr_before  # milestone not yet reached

        # Epoch 1
        train_one_epoch(
            s['model'], s['train_loader'], s['c6_loader'],
            s['loss_fn'], s['optimizer'], s['device'],
        )
        scheduler.step()  # last_epoch -> 2, milestone reached
        lr_after_epoch1 = s['optimizer'].param_groups[0]['lr']
        assert lr_after_epoch1 == pytest.approx(lr_before * 0.5)

    def test_early_stopping(self):
        from utils import EarlyStopping
        es = EarlyStopping(patience=2)
        assert es.step(1.0) is False
        assert es.step(2.0) is False
        assert es.step(3.0) is True

    def test_weighted_sum_loss_with_model_output(self, training_setup):
        """Bug T2/T5/M5: WeightedSumLoss works with real model forward output."""
        s = training_setup
        batch = next(iter(s['train_loader']))
        tau, logm, y_true, yATM = [b.clone().to(s['device']) for b in batch]

        c6_batch = next(iter(s['c6_loader']))
        tau_c6, logm_c6, yATM_c6 = [b.clone().to(s['device']) for b in c6_batch]

        out, g_tau, g_logm, g_logm2 = s['model'](tau, logm, yATM)
        out_c6, _, _, g_logm2_c6 = s['model'](tau_c6, logm_c6, yATM_c6)

        loss = s['loss_fn'](
            out, y_true, logm, g_tau, g_logm, g_logm2,
            out_c6, logm_c6, g_logm2_c6,
        )
        assert loss.dim() == 0
        assert loss.requires_grad
        loss.backward()
