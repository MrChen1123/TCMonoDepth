import os
import shutil

path = "/data/chenqilin/data/depth_datas/NYU_out/"

with open("./train_scenes.txt", "r") as f:
    train_scenes = f.readlines()

for train_scene in train_scenes:
    scene_path = os.path.join(path, train_scene.replace("\n", "") + "*")
    new_p = path + "train/"
    cmd = f"mv {scene_path} {new_p}"
    os.system(cmd)

print("done!")