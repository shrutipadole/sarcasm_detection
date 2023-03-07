from os import listdir
from os.path import isfile, join
import pandas as pd
df = pd.read_csv("data.csv")



path_to_list_of_files = "../data/source/list_of_files.txt"
dump_path = "../data/dataset/"

with open(path_to_list_of_files, "r") as f:
    list_of_main_files = f.readlines()

onlyfiles = [f for f in listdir(dump_path) if isfile(join(dump_path, f))]

for each_file in list_of_main_files:
    list_of_sub_files = [pd.read_csv(fle+".csv") for fle in onlyfiles if fle.startswith(each_file)]
    consolidated_df = pd.concat(list_of_sub_files)
    consolidated_df.to_csv("../data/clean_dataset/"+each_file+".csv")
