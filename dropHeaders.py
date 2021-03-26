import pathlib as pl
import pandas as pd

lam_directory = pl.Path(r"E:\Code_folder\DSS_all_Ctrl_dataset - Copy")
header_row = 2


def find_paths(lam_directory, header_row):
    file_paths = lam_directory.rglob('*.csv')
    exclude = lam_directory.joinpath("Analysis Data")
    return [p for p in file_paths if not str(p).startswith(str(exclude))]


def read_write(path):
    file = pd.read_csv(path, header=2)
    file.loc[:, ~file.columns.str.startswith("Unnamed")].to_csv(path, index=False)


for filepath in find_paths(lam_directory, header_row):
    read_write(filepath)

print("DONE")
