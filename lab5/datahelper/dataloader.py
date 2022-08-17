from torch.utils.data import DataLoader
from datahelper.dataset import IclevrDataset

def load_train_data(args):
    print("\nBuilding training & testing dataset...")

    train_dataset = IclevrDataset(args, "train")
    test_dataset = IclevrDataset(args, "test")

    print(f"# training samples: {len(train_dataset)}")
    print(f"# testing samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False
    )

    return train_loader, test_loader

def load_test_data(args):
    print("\nBuilding testing dataset...")

    test_dataset = IclevrDataset(args, "test")

    print(f"# testing samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False
    )

    return test_loader