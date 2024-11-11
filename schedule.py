from train_torch import TrainTorch

if __name__ == '__main__':

    path = "experiments"

    train = TrainTorch()

    experiments = [
        # Reduced CIFAR10
        {
            "dataset": "cifar_reduced",
            "loss": "loss_spring",
            "hash_size": 24,
            "margin": 2.0,
            "batch_size": 100,
            "total_epoch_count": 120,
            "number_of_epochs_per_decay": 100.0,
            "weight_decay_factor": 0.0002,
            "learning_rate": 0.07,
            "learning_rate_decay_factor": 2.0 / 3.0
        }
    ]

    for e in experiments:
        train.run(path, e)
