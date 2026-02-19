"""Transfer learning utilities for fine-tuning models on extended data.

Handles three scenarios:
1. Full transfer: all layers match (Base, DGM, HyperIV) — load all weights
2. Partial transfer: some layers changed size (Adjustment, DDPM) — load matching,
   re-initialize mismatched layers with Xavier init
3. Layer freezing: optionally freeze transferred layers for warmup epochs

Usage:
    from transfer import load_finetune_weights, setup_finetune_optimizer

    # Load weights (handles dimension mismatches automatically)
    transferred, reinitialized = load_finetune_weights(model, checkpoint_path, device)

    # Setup optimizer with lower LR for transferred layers, higher for new layers
    optimizer = setup_finetune_optimizer(model, transferred, reinitialized,
                                         base_lr=0.0002, new_lr=0.001)
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def load_finetune_weights(model, checkpoint_path, device='cpu'):
    """Load checkpoint weights into model, handling dimension mismatches.

    For layers where shapes match: copy weights directly.
    For layers where shapes differ: re-initialize with Xavier uniform.

    Args:
        model: the new model instance (potentially different architecture)
        checkpoint_path: path to the old .pt checkpoint (state_dict)
        device: torch device

    Returns:
        transferred: list of parameter names that were successfully loaded
        reinitialized: list of parameter names that were re-initialized
    """
    old_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    new_state = model.state_dict()

    transferred = []
    reinitialized = []
    skipped = []

    for name, new_param in new_state.items():
        if name in old_state:
            old_param = old_state[name]
            if old_param.shape == new_param.shape:
                # Exact match — transfer
                new_state[name] = old_param
                transferred.append(name)
            else:
                # Shape mismatch — partial transfer where possible, else re-init
                _partial_transfer(name, old_param, new_param, new_state)
                reinitialized.append(name)
                logger.info(f'  Re-initialized {name}: {old_param.shape} -> {new_param.shape}')
        else:
            # New parameter not in old checkpoint
            skipped.append(name)
            logger.info(f'  New parameter (not in checkpoint): {name} {new_param.shape}')

    model.load_state_dict(new_state)

    # Log old params not used
    unused = set(old_state.keys()) - set(new_state.keys())
    if unused:
        logger.info(f'  Unused checkpoint params: {unused}')

    logger.info(f'Transfer summary: {len(transferred)} transferred, '
                f'{len(reinitialized)} re-initialized, {len(skipped)} new')
    return transferred, reinitialized


def _partial_transfer(name, old_param, new_param, state_dict):
    """Try to partially transfer weights when dimensions differ.

    For weight matrices where only one dimension changed (e.g., input_dim expanded),
    copy the overlapping portion and Xavier-init the rest.
    """
    if old_param.dim() != new_param.dim():
        # Different number of dimensions — full re-init
        nn.init.xavier_uniform_(new_param) if new_param.dim() >= 2 else nn.init.zeros_(new_param)
        state_dict[name] = new_param
        return

    if old_param.dim() == 1:
        # Bias vector — copy overlapping portion
        min_size = min(old_param.shape[0], new_param.shape[0])
        new_param[:min_size] = old_param[:min_size]
        state_dict[name] = new_param
        return

    if old_param.dim() == 2:
        # Weight matrix (out_features, in_features)
        # Copy the overlapping block
        min_out = min(old_param.shape[0], new_param.shape[0])
        min_in = min(old_param.shape[1], new_param.shape[1])

        # Xavier init the whole thing first, then overwrite the overlap
        nn.init.xavier_uniform_(new_param)
        new_param[:min_out, :min_in] = old_param[:min_out, :min_in]
        state_dict[name] = new_param
        return

    # Higher-dimensional tensors (e.g., conv weights): full re-init
    if new_param.dim() >= 2:
        nn.init.xavier_uniform_(new_param.view(new_param.shape[0], -1))
        new_param = new_param.view_as(state_dict[name])
    state_dict[name] = new_param


def setup_finetune_optimizer(model, transferred, reinitialized,
                              base_lr=0.0001, new_lr=0.001, weight_decay=1e-5):
    """Create optimizer with differential learning rates.

    Transferred layers get lower LR (fine-tuning), re-initialized layers get higher LR.

    Args:
        model: the model
        transferred: list of param names that were loaded from checkpoint
        reinitialized: list of param names that were re-initialized
        base_lr: learning rate for transferred (pre-trained) parameters
        new_lr: learning rate for new/re-initialized parameters
        weight_decay: L2 regularization

    Returns:
        torch.optim.AdamW optimizer with parameter groups
    """
    transferred_set = set(transferred)
    reinitialized_set = set(reinitialized)

    pretrained_params = []
    new_params = []

    for name, param in model.named_parameters():
        if name in transferred_set:
            pretrained_params.append(param)
        else:
            new_params.append(param)

    param_groups = [
        {'params': pretrained_params, 'lr': base_lr, 'weight_decay': weight_decay},
        {'params': new_params, 'lr': new_lr, 'weight_decay': weight_decay},
    ]

    optimizer = torch.optim.AdamW(param_groups)
    logger.info(f'Fine-tune optimizer: {len(pretrained_params)} pretrained params (lr={base_lr}), '
                f'{len(new_params)} new params (lr={new_lr})')
    return optimizer


def freeze_transferred(model, transferred, freeze=True):
    """Freeze or unfreeze transferred parameters.

    Use this for warmup: freeze pretrained layers for a few epochs
    to let new layers catch up, then unfreeze everything.

    Args:
        model: the model
        transferred: list of param names to freeze/unfreeze
        freeze: True to freeze, False to unfreeze
    """
    transferred_set = set(transferred)
    count = 0
    for name, param in model.named_parameters():
        if name in transferred_set:
            param.requires_grad = not freeze
            count += 1

    state = 'Frozen' if freeze else 'Unfrozen'
    logger.info(f'{state} {count} transferred parameters')
