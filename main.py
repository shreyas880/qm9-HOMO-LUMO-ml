import os
from rdkit import Chem
# from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflow as tf
from sklearn.model_selection import train_test_split
from featurize import featurize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Draw
import joblib
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

MODEL_PATH = "rf_qm9_model.pkl"


df = pd.read_csv('qm9_dataset.csv')

ds = df[:50000]

features = []
targets = []

for _, row in ds.iterrows():
    fp = featurize(row['smiles'])
    if fp is not None:
        features.append(fp)
        targets.append(row['gap'])

X = np.array(features)
y = np.array(targets)

bit_effects = []

for bit in range(X.shape[1]):

    present = y[X[:, bit] == 1]
    absent = y[X[:, bit] == 0]

    if len(present) < 20 or len(absent) < 20:
        continue

    mean_present = present.mean()
    mean_absent = absent.mean()

    diff = mean_present - mean_absent

    bit_effects.append((bit, diff, mean_present, mean_absent, len(present)))

effects_df = pd.DataFrame(
    bit_effects,
    columns=["bit", "gap_difference", "gap_when_present", "gap_when_absent", "count"]
)

os.makedirs("model_results", exist_ok=True)
effects_df.to_csv("model_results/bit_effects.csv", index=False)


bit_effects = sorted(bit_effects, key=lambda x: abs(x[1]), reverse=True)

top_bits = [b[0] for b in bit_effects[:7]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)


if os.path.exists(MODEL_PATH):

    print("Loading saved model...")
    model = joblib.load(MODEL_PATH)

else:

    print("Training new model...")
    model = RandomForestRegressor(
        n_estimators=100,
        criterion='squared_error',
        min_samples_split=5,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print("Model saved to", MODEL_PATH)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("MAE:", mae)
print("R2:", r2)

# plt.scatter(y_test, pred, alpha=0.4)
# plt.xlabel('True HOMO-LUMO Gap')
# plt.ylabel('Predicted HOMO-LUMO Gap')
# x, y = [0, 0.45], [0, 0.45]
# plt.plot(x, y)
# plt.show()

# print("\nMost influential substructures:\n")
# for bit, diff, mp, ma, count in bit_effects[:10]:

#     direction = "increases" if diff > 0 else "decreases"

#     print(
#         f"Bit {bit}: {direction} gap by {abs(diff):.3f} eV "
#         f"(present={mp:.3f}, absent={ma:.3f}, n={count})"
#     )

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

found = {bit: [] for bit in top_bits}
MAX_MOLS_PER_BIT = 10

for smiles in ds["smiles"]:

    fp, mol, bitInfo = featurize(smiles, return_bitinfo=True)

    if fp is None:
        continue

    active_bits = np.where(fp == 1)[0]

    for bit in top_bits:

        if bit in active_bits and len(found[bit]) < MAX_MOLS_PER_BIT:
            found[bit].append((mol, bitInfo))

    if all(len(found[bit]) >= MAX_MOLS_PER_BIT for bit in top_bits):
        break

for bit, data_list in found.items():

    if len(data_list) == 0:
        continue

    bit_dir = os.path.join("imp_bit_png", f"fp_bit_{bit}")
    os.makedirs(bit_dir, exist_ok=True)

    for i, (mol, bitInfo) in enumerate(data_list):

        atom_id, radius = bitInfo[bit][0]

        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_id)

        atoms = set()
        for bond_idx in env:
            bond = mol.GetBondWithIdx(bond_idx)
            atoms.add(bond.GetBeginAtomIdx())
            atoms.add(bond.GetEndAtomIdx())

        img = Draw.MolToImage(
            mol,
            highlightAtoms=list(atoms),
            legend=f"Bit {bit}",
            size=(400,300)
        )

        # img = Draw.MolToImage(mol, highlightAtoms=list(atoms))

        path = os.path.join(bit_dir, f"example_{i}.png")
        img.save(path)

for bit in top_bits:

    bit = int(bit)  # ensure scalar integer

    mask = X[:, bit] == 1

    present = y[mask]
    absent = y[~mask]
    plt.figure(figsize=(6,4))

    plt.hist(present, bins=40, alpha=0.6, label="bit present")
    plt.hist(absent, bins=40, alpha=0.6, label="bit absent")

    plt.xlabel("HOMO-LUMO Gap")
    plt.ylabel("Molecule count")
    plt.title(f"Effect of fingerprint bit {bit}")

    plt.legend()

    plt.savefig(f"model_results/bit_{bit}_distribution.png", dpi=300)
    plt.close()

with open("model_results/model_summary.txt", "w") as f:

    f.write("QM9 HOMO-LUMO Prediction\n\n")

    f.write(f"MAE: {mae}\n")
    f.write(f"R2: {r2}\n\n")

    f.write("Top fingerprint bits influencing gap:\n")

    for bit, diff, mp, ma, count in bit_effects[:10]:

        direction = "increase" if diff > 0 else "decrease"

        f.write(
            f"Bit {bit}: {direction} gap by {abs(diff):.4f} eV "
            f"(present={mp:.4f}, absent={ma:.4f}, n={count})\n"
        )

# Smiles - 1, Homo - 2, Lumo - 3, Gap - 4