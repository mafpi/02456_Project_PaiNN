"""
Basic example of how to train the PaiNN model to predict the QM9 property
"internal energy at 0K". This property (and the majority of the other QM9
properties) is computed as a sum of atomic contributions.
"""
import torch
import argparse
from tqdm import trange
import torch.nn.functional as F
from src.data import QM9DataModule
from pytorch_lightning import seed_everything
from src.models import PaiNN, AtomwisePostProcessing
from matplotlib import pyplot as plt
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
import numpy as np

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0)

    # Data
    parser.add_argument('--target', default=7, type=int) # 7 => Internal energy at 0K
    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument('--batch_size_train', default=100, type=int)
    parser.add_argument('--batch_size_inference', default=1000, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--splits', nargs=3, default=[110000, 10000, 10831], type=int) # [num_train, num_val, num_test]
    parser.add_argument('--subset_size', default=None, type=int)

    # Model
    parser.add_argument('--num_message_passing_layers', default=10, type=int)
    parser.add_argument('--num_features', default=128, type=int)
    parser.add_argument('--num_outputs', default=1, type=int)
    parser.add_argument('--num_rbf_features', default=20, type=int)
    parser.add_argument('--num_unique_atoms', default=100, type=int)
    parser.add_argument('--cutoff_dist', default=5.0, type=float)

    # Training
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--patience', default=30, type=int)
    parser.add_argument('--swa_lr', default=0.0001, type=float)

    args = parser.parse_args()
    return args


def run_experiment(args):
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dm = QM9DataModule(
        target=args.target,
        data_dir=args.data_dir,
        batch_size_train=args.batch_size_train,
        batch_size_inference=args.batch_size_inference,
        num_workers=args.num_workers,
        splits=args.splits,
        seed=args.seed,
        subset_size=args.subset_size,
    )
    dm.prepare_data()
    dm.setup()
    y_mean, y_std, atom_refs = dm.get_target_stats(
        remove_atom_refs=True, divide_by_atoms=True
    )

    painn = PaiNN(
        num_message_passing_layers=args.num_message_passing_layers,
        num_features=args.num_features,
        num_outputs=args.num_outputs,
        num_rbf_features=args.num_rbf_features,
        num_unique_atoms=args.num_unique_atoms,
        cutoff_dist=args.cutoff_dist,
    ).to(device)

    # Load the pre-trained model weights
    painn.load_state_dict(torch.load("Results/best_model.pth", map_location = device))

    post_processing = AtomwisePostProcessing(
        args.num_outputs, y_mean, y_std, atom_refs
    ).to(device)

    # Define optimizer
    optimizer = optim.SGD(painn.parameters(), lr=args.swa_lr, momentum=0.9)

    # Wrap PaiNN with SWA
    swa_model = AveragedModel(painn).to(device)

    train_losses = []

    # Training Loop
    painn.train()
    pbar = trange(50)
    for epoch in pbar:
        loss_epoch = 0.
        for batch in dm.train_dataloader():
            batch = batch.to(device)

            atomic_contributions = painn(
                atoms=batch.z,
                atom_positions=batch.pos,
                graph_indexes=batch.batch
            )
            preds = post_processing(
                atoms=batch.z,
                graph_indexes=batch.batch,
                atomic_contributions=atomic_contributions,
            )

            loss_step = F.mse_loss(preds, batch.y, reduction='sum')
            loss = loss_step / len(batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss_step.detach().item()
        loss_epoch /= len(dm.data_train)
        train_losses.append(loss_epoch)

        pbar.set_postfix_str(f'Train loss: {loss_epoch:.3e}')

        swa_model.update_parameters(painn)

    losses = {'train_losses': train_losses}
    torch.save(losses, "swa_losses.pt")
    # Evaluate SWA model
    swa_model.eval()
    mae = 0
    with torch.no_grad():
        for batch in dm.test_dataloader():
            batch = batch.to(device)

            atomic_contributions = swa_model(
                atoms=batch.z,
                atom_positions=batch.pos,
                graph_indexes=batch.batch,
            )
            preds = post_processing(
                atoms=batch.z,
                graph_indexes=batch.batch,
                atomic_contributions=atomic_contributions,
            )
            mae += F.l1_loss(preds, batch.y, reduction='sum')

    mae /= len(dm.data_test)
    unit_conversion = dm.unit_conversion[args.target]
    test_mae = unit_conversion(mae)
    print(f'Test MAE (SWA): {test_mae:.3f}')

    # Save to a text file
    output_text = f'Test MAE: {test_mae:.3f}'
    with open(f"test_results_swa_{args.swa_lr}lr.txt", 'w') as file:
        file.write(output_text)

    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(f'loss_plot_swa_{args.swa_lr}lr.png')
    plt.show()

    return test_mae

if __name__ == '__main__':
    args = cli()
    learning_rates = [10**-7]
    results = []

    for lr in learning_rates:
        args.swa_lr = lr
        test_mae = run_experiment(args)
        results.append((lr, test_mae))

    # Save results to a text file
    with open("swa_lr_vs_mae.txt", 'w') as file:
        for lr, mae in results:
            file.write(f"SWA Learning Rate: {lr} Test MAE: {mae}\n")

    # Plot the results
    lrs, maes = zip(*results)
    plt.plot(lrs, maes, marker='o')
    plt.xlabel("SWA Learning Rate")
    plt.ylabel("Test MAE")
    plt.title("Test MAE vs SWA Learning Rate")
    plt.xscale('log')
    plt.savefig('swa_lr_vs_mae.png')
    plt.show()