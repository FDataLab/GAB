if __name__ == '__main__':
    import sys
    import os
    
    # Edit path to import from different module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from data_loader import DataLoader
    from utility.config import args
    from defense_model.improving_graph import GarnetPurification


    args.dataset = "cora"
    data,split = DataLoader.load(args.dataset)

    filter = GarnetPurification()
    print(data.edge_index.shape)
    data = filter(data)
    print(data.edge_index.shape)
