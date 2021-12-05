import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from os import listdir
from os.path import isfile, join

trg_path = f"datasets/nuscenes_pred/label/train"
val_path = f"datasets/nuscenes_pred/label/val"
test_path = f"datasets/nuscenes_pred/label/test"

def get_all_scenes(path):
    scene_names = [f[:-4] for f in listdir(path) if isfile(join(path, f))]
    return scene_names

def create_df(path, scene_name):
    scene_file = f"{path}/{scene_name}.txt"
    scene_num = int(scene_name.split("-")[-1])
    prefix = str(scene_num)

    data = []
    with open(scene_file, "r") as f:

        while r := f.readline():
            temp_ls = r.split(" ")
            temp_ls = temp_ls[0:2] + [temp_ls[13]] + [temp_ls[15]] + [temp_ls[14]] + [temp_ls[16]] 
            data.append(temp_ls)
        
    df_raw = pd.DataFrame(data, columns=["frame_id", "instance_id", "x", "y", "z", "rotation"])
    df_raw["instance_id"] = prefix + df_raw["instance_id"]

    return df_raw.astype(float)

def normalize_df(df_raw):
    sc = MinMaxScaler()
    new_df = df_raw.iloc[:,2:]
    new_df_names = df_raw.columns[2:]
    df_scaled = sc.fit_transform(new_df.values)
    df = pd.DataFrame(df_scaled, columns=new_df_names)
    df.insert(0, "frame_id", df_raw["frame_id"])
    df.insert(1, "instance_id", df_raw["instance_id"])

    return df

def get_uids(df):
    inst_ids =  np.unique(df["instance_id"].values)
    return inst_ids

def create_df_dict(df, inst_ids):
    df_dict = {}
    for i in inst_ids:
        df_dict[i] = df.loc[df["instance_id"] == i]
    return df_dict

def get_sorted_df(scenes_dict):

    scenes_df_dict = {}
    for key, df_dict in scenes_dict.items():
        scenes_df_dict[key] = pd.concat([df_dict[key] for key in df_dict.keys()])
    
    sorted_df = pd.concat([scenes_df_dict[key] for key in scenes_df_dict.keys()])

    return sorted_df

def export_dataset(scene_names, datatype):

    if datatype == "trg":
        path = trg_path
    elif datatype == "val":
        path = val_path
    elif datatype == "test":
        path = test_path
    else:
        return "Invalid path!"

    scenes_dict = {}
    for scene_name in scene_names:
        df_raw = create_df(path, scene_name)
        # df = normalize_df(df_raw)
        inst_ids = get_uids(df_raw)
        df_dict = create_df_dict(df_raw, inst_ids)
        scenes_dict[scene_name] = df_dict

    sorted_df = get_sorted_df(scenes_dict)
    print(sorted_df)
    sorted_df.to_csv(f"./csv/{datatype}1.csv", index=False)

    return scenes_dict, sorted_df

def plot_multiple_scenes(scenes_dict, scene_names):
    ### scenes_dict: {<scene_name> : {<instance_id> : <dataframe>}}
    ### scene_names: List of scene_names

    fig, ax = plt.subplots()
    for scene_name in scene_names:     
        for key in scenes_dict[scene_name].keys():
            ax.scatter(scenes_dict[scene_name][key]["x"], scenes_dict[scene_name][key]["y"], s=10, alpha=0.7, label=f"{scene_name}-{key}")

    ax.legend()    
    plt.show()

if __name__ == "__main__":

    trg_scene_names = get_all_scenes(trg_path)
    val_scene_names = get_all_scenes(val_path)
    test_scene_names = get_all_scenes(test_path)
    print(len(trg_scene_names))
    print(len(val_scene_names))
    print(len(test_scene_names))

    datatype = "test"

    scenes_dict, sorted_df = export_dataset(test_scene_names, datatype)
    # plot_multiple_scenes(scenes_dict, trg_scene_names)