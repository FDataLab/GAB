if __name__ == "__main__":
    import os
    import sys

    # Edit path to import from different module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from data_loader import DataLoader
    from defense_model.improving_graph import GarnetPurification
    from utility.config import args

    args.dataset = "cora"
    data, split = DataLoader.load(args.dataset)

    filter = GarnetPurification()
    print(data.edge_index.shape)
    data = filter(data)
    print(data.edge_index.shape)
