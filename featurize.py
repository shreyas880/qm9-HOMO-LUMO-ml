from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=1024
)

def featurize(smiles, return_bitinfo=False):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        if return_bitinfo:
            return None, None
        return None

    if return_bitinfo:
        bitInfo = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=1024,
            bitInfo=bitInfo
        )
        return np.array(fp), mol, bitInfo

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=1024
    )

    return np.array(fp)