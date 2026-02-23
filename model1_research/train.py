import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model import MultiModel, WeightedSumLoss
from dataset import DataProcessor
from utils import load_config, parse_list_config, parse_date, set_seed, setup_logging, MetricsTracker, EarlyStopping

from argparse import ArgumentParser

import torch
from torch import optim
from tqdm import tqdm

import os
import matplotlib.pyplot as plt


def train_one_epoch(model, train_loader, c6_loader, loss_function, optimizer, device, gradient_clip=1.0):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    c6_iterator = iter(c6_loader)
    train_bar = tqdm(train_loader, desc="Training")

    for tau_train, logm_train, y_train, yATM_train in train_bar:
        try:
            tau_syn, logm_syn, yATM_syn = next(c6_iterator)
        except StopIteration:
            c6_iterator = iter(c6_loader)
            tau_syn, logm_syn, yATM_syn = next(c6_iterator)

        tau_train = tau_train.to(device)
        logm_train = logm_train.to(device)
        y_train = y_train.to(device)
        yATM_train = yATM_train.to(device)
        tau_syn = tau_syn.to(device)
        logm_syn = logm_syn.to(device)
        yATM_syn = yATM_syn.to(device)

        output_train, grad_ttm1_train, grad_logm1_train, grad_logm2_train = model(tau_train, logm_train, yATM_train)
        output_syn, _, _, grad_logm2_syn = model(tau_syn, logm_syn, yATM_syn)

        # Bug T5/M5 fixed: loss_function now returns single tensor
        loss = loss_function(
            output_train, y_train, logm_train,
            grad_ttm1_train, grad_logm1_train, grad_logm2_train,
            output_syn, logm_syn, grad_logm2_syn
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        train_bar.set_postfix(loss=f'{loss.item():.6f}')

    return total_loss / max(n_batches, 1)


def validate(model, val_loader, c6_loader, loss_function, device):
    """Validate model. Returns average loss.

    Note: SmileModel.forward() uses torch.autograd.grad(create_graph=True),
    so we cannot wrap the forward pass in torch.no_grad(). Instead we run
    the forward pass with gradients enabled, then detach the loss.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    c6_iterator = iter(c6_loader)

    for tau_val, logm_val, y_val, yATM_val in val_loader:
        try:
            tau_syn, logm_syn, yATM_syn = next(c6_iterator)
        except StopIteration:
            c6_iterator = iter(c6_loader)
            tau_syn, logm_syn, yATM_syn = next(c6_iterator)

        tau_val = tau_val.to(device)
        logm_val = logm_val.to(device)
        y_val = y_val.to(device)
        yATM_val = yATM_val.to(device)
        tau_syn = tau_syn.to(device)
        logm_syn = logm_syn.to(device)
        yATM_syn = yATM_syn.to(device)

        # Forward pass needs autograd for SmileModel gradient computation
        output_val, grad_ttm1_val, grad_logm1_val, grad_logm2_val = model(tau_val, logm_val, yATM_val)
        output_syn, _, _, grad_logm2_syn = model(tau_syn, logm_syn, yATM_syn)

        with torch.no_grad():
            loss = loss_function(
                output_val, y_val, logm_val,
                grad_ttm1_val, grad_logm1_val, grad_logm2_val,
                output_syn, logm_syn, grad_logm2_syn
            )
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = ArgumentParser()
    parser.add_argument("--on_gpu", action='store_true')
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--finetune", type=str, default=None,
                        help='Path to checkpoint for fine-tuning (transfer learning)')
    args = parser.parse_args()

    config = load_config('config.ini')

    # Seed
    seed = config['training'].getint('seed')
    set_seed(seed)

    # Logging
    log_dir = config['save_path']['log_dir']
    logger = setup_logging(log_dir, 'base_model')

    # Device
    use_gpu = torch.cuda.is_available() and args.on_gpu
    logger.info(f'Use GPU: {use_gpu}')
    device = torch.device("cuda:0" if use_gpu else "cpu")

    # Dates
    train_start_date = parse_date(config['training']['train_start_date'])
    train_end_date = parse_date(config['training']['train_end_date'])

    # Data
    logger.info('Preprocessing...')
    dp = DataProcessor(config)
    dp()

    batch_size = config['training'].getint('batch_size')
    # Bug T1/D5 fixed: call Prepare_train_data (not Prepare_prs_dataset)
    # Bug T3 fixed: returns DataLoaders
    train_loader, val_loader, c6_loader = dp.Prepare_train_data(train_start_date, train_end_date, batch_size)
    logger.info('End of data preprocessing.')

    # Model
    logger.info('Model generating...')
    learning_rate = config['model_sett'].getfloat('learning_rate')
    ensemble_num = config['model_sett'].getint('ensemble_num')
    hidden_sizes = [int(x) for x in parse_list_config(config['model_sett']['hidden_sizes'], int)]
    loss_weights = parse_list_config(config['model_sett']['loss_weights'])

    torch.set_default_dtype(torch.float64)
    model = MultiModel(hidden_sizes=hidden_sizes, ensemble_num=ensemble_num).to(device)
    # Bug T2 fixed: pass weights list instead of device
    loss_function = WeightedSumLoss(weights=loss_weights).to(device)

    # Fine-tuning: load pretrained weights with differential LR
    if args.finetune and os.path.exists(args.finetune):
        from transfer import load_finetune_weights, setup_finetune_optimizer
        logger.info(f'Fine-tuning from: {args.finetune}')
        transferred, reinitialized = load_finetune_weights(model, args.finetune, device)
        optimizer = setup_finetune_optimizer(model, transferred, reinitialized,
                                             base_lr=learning_rate * 0.1, new_lr=learning_rate)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    epochs = args.epochs if args.epochs is not None else config['training'].getint('epochs')
    milestone_start = config['training'].getint('scheduler_milestones_start')
    milestone_step = config['training'].getint('scheduler_milestones_step')
    scheduler_gamma = config['training'].getfloat('scheduler_gamma')
    gradient_clip = config['training'].getfloat('gradient_clip')

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[i for i in range(milestone_start, epochs, milestone_step)],
        gamma=scheduler_gamma
    )

    # Tracking
    metrics = MetricsTracker()
    patience = config['training'].getint('early_stopping_patience')
    early_stopping = EarlyStopping(patience=patience)

    logger.info('Training...')
    best_loss = config['model_sett'].getfloat('best_loss')
    model_path = config['save_path']['model_path']
    plot_path = config['save_path']['plot_path']
    os.makedirs(plot_path, exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, c6_loader, loss_function, optimizer, device, gradient_clip)
        val_loss = validate(model, val_loader, c6_loader, loss_function, device)

        # Bug T4 fixed: scheduler.step() in epoch loop, not batch loop
        scheduler.step()

        is_best = metrics.update(epoch, train_loss, val_loss)
        logger.info(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - LR: {scheduler.get_last_lr()[0]:.2e}')

        if is_best and val_loss < best_loss:
            best_loss = val_loss
            config['model_sett']['best_loss'] = str(best_loss)
            with open('config.ini', 'w') as configfile:
                config.write(configfile)
            torch.save(model.state_dict(), model_path)
            logger.info(f'  New best model saved (val_loss={val_loss:.6f})')

        if early_stopping.step(val_loss):
            logger.info(f'Early stopping at epoch {epoch+1}')
            break

    # Save metrics
    metrics.save(os.path.join(log_dir, 'metrics.json'))

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(metrics.train_losses, label='Training Loss')
    plt.plot(metrics.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    plt.savefig(os.path.join(plot_path, f'training_loss_{epochs}epochs.png'), dpi=150, bbox_inches='tight')
    plt.close()

    logger.info('Finished Training')


if __name__ == '__main__':
    main()
