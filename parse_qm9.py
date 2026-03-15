import os
import pandas as pd

folder = r"C:\Users\Shreyas\datasets\QM9"

data = []

files = sorted([
    f for f in os.listdir(folder)
    if f.endswith(".xyz") and os.path.isfile(os.path.join(folder, f))
])

print("Total valid molecule files:", len(files))

for file in files:

    path = os.path.join(folder, file)

    try:
        with open(path) as f:
            lines = f.readlines()

        if len(data) % 5000 == 0:
            print("Processed:", len(data))

        properties = lines[1].split()

        homo = float(properties[7])
        lumo = float(properties[8])
        gap = lumo - homo

        smiles = lines[-2].strip()

        data.append({
            "molecule": file,
            "smiles": smiles,
            "homo": homo,
            "lumo": lumo,
            "gap": gap
        })

    except Exception as e:
        print("Skipping file:", file, "| Error:", e)


df = pd.DataFrame(data)

df.to_csv("qm9_dataset.csv", index=False)

print("Dataset saved!")