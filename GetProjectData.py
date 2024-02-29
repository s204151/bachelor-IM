import pandas as pd
import numpy as np

# Load data
abs_path = r"Data for thesis\abs.xlsx"
idpe_path = r"Data for thesis\ldpe.xlsx"
moplen_path = r"Data for thesis\moplen.xlsx"
paths = [abs_path, idpe_path, moplen_path]

# material index definitions:
mat_abs = 0
mat_ldpe = 1
mat_moplen = 2

def get_data():
    data_material = []
    for i in range(3):
        temp_data = pd.read_excel(paths[i])
        data_material.append(temp_data)
        temp_data["material"] = [i] * temp_data.shape[0]

    return data_material

    # excel_data = pd.DataFrame()
    # melt_temp_idx = 0
    # pack_press_idx = 1
    # inj_spd_idx = 2

def get_concat_data():
    materials = ["ABS", "IDPE", "Moplen"]

    data_material = []
    excel_data = pd.DataFrame()
    for i in range(3):
        temp_data = pd.read_excel(paths[i])
        data_material.append(temp_data)
        temp_data["material"] = materials[i]
        # temp_data["material"] = i

        excel_data = pd.concat([excel_data, temp_data])

    excel_data = excel_data.drop("rand", axis=1)
    return excel_data