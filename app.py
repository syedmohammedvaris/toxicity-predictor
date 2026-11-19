import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="In Silico Toxicity Predictor", page_icon="🧪", layout="wide")

TRAINING_DATA = [
    ("CC(=O)OC1=CC=CC=C1C(=O)O", 0),       # Aspirin
    ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", 0),   # Ibuprofen
    ("CC(=O)NC1=CC=C(C=C1)O", 0),           # Acetaminophen
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 0),    # Caffeine
    ("C(C1C(C(C(C(O1)O)O)O)O)O", 0),        # Glucose
    ("CCO", 0),                             # Ethanol
    ("CC(=O)O", 0),                         # Acetic acid
    ("C1=CC=C(C=C1)C(=O)O", 0),             # Benzoic acid
    ("CC(C)(C)NCC(O)C1=CC(CO)=C(O)C=C1", 0),# Salbutamol
    ("CN1CCCC1C2=CN=CC=C2", 1),             # Nicotine
    ("C1=CC=C(C=C1)[N+](=O)[O-]", 1),       # Nitrobenzene
    ("C1=CC=C2C(=C1)C=CC3=CC=CC=C32", 1),   # Phenanthrene
    ("C1=CC2=C3C(=C1)C=CC4=CC=CC(=C43)C=C2", 1), # Pyrene
    ("ClC1=CC=C(C=C1)C(C2=CC=C(Cl)C=C2)C(Cl)(Cl)Cl", 1), # DDT
    ("C1=CC=C(C=C1)N=NC2=CC=CC=C2", 1),     # Azobenzene
    ("C1=CC=C(C=C1)Cl", 1),                 # Chlorobenzene
    ("C1=CC=C2C(=C1)C=CC=N2", 1),           # Quinoline
    ("CC(=O)NC1=CC=C(C=C1)C2=CC=CC=C2", 1), # 4-Acetylaminobiphenyl
    ("C1=CC(=CC=C1N)N", 1),                 # p-Phenylenediamine
]

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
    return np.array(fp)

@st.cache_resource
def load_model():
    X_train = np.array([smiles_to_fp(s) for s, _ in TRAINING_DATA])
    y_train = np.array([label for _, label in TRAINING_DATA])
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

model = load_model()

st.title("🧪 In Silico Chemical Toxicity Prediction Platform")
st.write("Enter a chemical **SMILES** string or pick an example molecule to run real-time QSAR toxicity prediction and molecular profiling.")

examples = {
    "Aspirin (Low Risk)": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine (Low Risk)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Nitrobenzene (High Risk)": "C1=CC=C(C=C1)[N+](=O)[O-]",
    "Nicotine (High Risk)": "CN1CCCC1C2=CN=CC=C2",
    "DDT Pesticide (High Risk)": "ClC1=CC=C(C=C1)C(C2=CC=C(Cl)C=C2)C(Cl)(Cl)Cl"
}

col1, col2 = st.columns([1, 1])

with col1:
    selected_example = st.selectbox("Quick Example Molecules:", list(examples.keys()))
    smiles_input = st.text_input("Or Enter Custom SMILES String:", value=examples[selected_example])
    run_btn = st.button("🔬 Run Toxicity Prediction", type="primary")

mol = Chem.MolFromSmiles(smiles_input.strip())

if mol is None:
    st.error("❌ Invalid SMILES string. Please check the chemical notation.")
else:
    fp = smiles_to_fp(smiles_input.strip()).reshape(1, -1)
    prob_toxic = float(model.predict_proba(fp)[0][1])
    risk_percent = round(prob_toxic * 100, 1)

    mw = round(Descriptors.MolWt(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = round(Descriptors.TPSA(mol), 2)

    with col2:
        if risk_percent >= 65:
            st.error(f"### 🔴 HIGH TOXICITY RISK ({risk_percent}%)")
        elif risk_percent >= 35:
            st.warning(f"### 🟡 MODERATE TOXICITY RISK ({risk_percent}%)")
        else:
            st.success(f"### 🟢 LOW TOXICITY RISK ({risk_percent}%)")
        
        img = Draw.MolToImage(mol, size=(320, 320))
        st.image(img, caption="2D Chemical Structure")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Predicted Endpoint Profile")
        hepato = "Elevated" if (logp > 3.0 and mw > 250) or risk_percent > 60 else "Low"
        mutagen = "Alert (Aromatic/Nitro)" if ("N(=O)=O" in smiles_input or "[N+](=O)[O-]" in smiles_input or risk_percent > 65) else "Negative"
        herg = "Moderate/High" if (logp > 3.5 and hba >= 2) else "Low"
        st.write(f"- **QSAR Toxicity Probability:** `{risk_percent}%`")
        st.write(f"- **Hepatotoxicity (Liver):** {hepato}")
        st.write(f"- **Ames Mutagenicity:** {mutagen}")
        st.write(f"- **Cardiotoxicity (hERG Channel):** {herg}")

    with c2:
        st.subheader("Physicochemical Properties")
        st.write(f"- **Molecular Weight:** `{mw} g/mol`")
        st.write(f"- **LogP (Lipophilicity):** `{logp}`")
        st.write(f"- **TPSA:** `{tpsa} Å²`")
        st.write(f"- **H-Bond Donors / Acceptors:** `{hbd}` / `{hba}`")
