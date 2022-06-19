#%% Replace Names
import os
counter = 0
path = r"C:\Users\DeepWorkspace\Desktop\KL Data\kneeKL299\auto_test"

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".png"):
            if file.find("_2") > -1:
                counter = counter + 1
                os.rename(os.path.join(root, file), os.path.join(root, file.replace("_2", "L")))
            if file.find("_1") > -1:
                counter = counter + 1
                os.rename(os.path.join(root, file), os.path.join(root, file.replace("_1", "R")))
    if counter == 0:
        print("No file has been found")