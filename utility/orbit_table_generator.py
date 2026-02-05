import os
import numpy as np
import pandas as pd
import os
import sys


class OrbitTableGenerator:
    def __init__(self,dataset):
        self.dataset = dataset
        self.filepath = 'data/orbit/'

    def generate_orbit_table(self):
        if self.dataset in['cora','citeseer','polblogs','pubmed'] :
            print("Loaded existing orbit table")
            if os.path.exists("{}{}_orbit_table.csv".format(self.filepath,self.dataset)):
                return pd.read_csv("{}{}_orbit_table.csv".format(self.filepath,self.dataset))
            return self.generate_orbit_tables_from_sratch()
        else:
            raise Exception("Unsupport dataset")

    def generate_orbit_tables_from_sratch(self):
        filename = self.filepath +"dpr_"+ self.dataset + '.out'
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Process the lines to create a list of lists
        data_list = [list(map(int, line.split())) for line in lines]

        # Convert the list of lists to a NumPy array

        graphlet_features = np.array(data_list)
        mylist = []
        for i in range(len(graphlet_features)):
            arr = graphlet_features[i]
            sorted_indices = np.argsort(arr)[::-1]
            if sorted_indices[0] < sorted_indices[1]:
                s1 = str(sorted_indices[0]) + str(sorted_indices[1])
            else:
                s1 = str(sorted_indices[1]) + str(sorted_indices[0])

            mylist.append([i, str(sorted_indices[0]), str(sorted_indices[1]), s1])

        my_array = np.array(mylist)
        df_2d = pd.DataFrame(my_array, columns=['node_number', 'Orbit_type_I', 'Orbit_type_II', 'two_Orbit_type'])
        df_2d.to_csv("{}{}_orbit_table.csv".format(self.filepath,self.dataset))
        return df_2d

def generate_orbit_tables_from_count(data_list,nodes_list):
    # Convert the list of lists to a NumPy array

    graphlet_features = np.array(data_list)
    mylist = []
    for i in range(len(graphlet_features)):
        arr = graphlet_features[i]
        sorted_indices = np.argsort(arr)[::-1]
        if sorted_indices[0] < sorted_indices[1]:
            s1 = str(sorted_indices[0]) + str(sorted_indices[1])
        else:
            s1 = str(sorted_indices[1]) + str(sorted_indices[0])

        mylist.append([nodes_list[i], str(sorted_indices[0]), str(sorted_indices[1]), s1])

    my_array = np.array(mylist)
    df_2d = pd.DataFrame(my_array, columns=['node_number', 'Orbit_type_I', 'Orbit_type_II', 'two_Orbit_type'])
    return df_2d

if __name__ == "__main__":
    OrbitTableGenerator("citeseer").generate_orbit_table()