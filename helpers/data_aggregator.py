from os import listdir
from os.path import isfile, join



path_to_list_of_files = "../data/source/list_of_files.txt"
dump_path = "../data/dataset/"

with open(path_to_list_of_files, "r") as f:
    list_of_main_files = f.readlines()

onlyfiles = [f for f in listdir(dump_path) if isfile(join(dump_path, f))]
