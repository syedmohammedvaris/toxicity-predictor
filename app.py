import streamlit as st
import numpy as np
import pandas as pd
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

def analyze_molecule(smiles):
    smiles = str(smiles).strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = smiles_to_fp(smiles).reshape(1, -1)
    prob_toxic = float(model.predict_proba(fp).item(1))
    risk_percent = round(prob_toxic * 100, 1)

    mw = round(Descriptors.MolWt(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = round(Descriptors.TPSA(mol), 2)

    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    lipinski_status = f"Pass ({violations} violations)" if violations <= 1 else f"Fail ({violations} violations)"

    if risk_percent >= 65:
        risk_label = "High Risk"
    elif risk_percent >= 35:
        risk_label = "Moderate Risk"
    else:
        risk_label = "Low Risk"

    hepato = "Elevated" if (logp > 3.0 and mw > 250) or risk_percent > 60 else "Low"
    mutagen = "Alert (Aromatic/Nitro)" if ("N(=O)=O" in smiles or "[N+](=O)[O-]" in smiles or risk_percent > 65) else "Negative"
    herg = "Moderate/High" if (logp > 3.5 and hba >= 2) else "Low"

    return {
        "mol": mol,
        "SMILES": smiles,
        "Toxicity_Probability_%": risk_percent,
        "Risk_Level": risk_label,
        "Hepatotoxicity": hepato,
        "Ames_Mutagenicity": mutagen,
        "hERG_Cardiotoxicity": herg,
        "Lipinski_Rule_of_5": lipinski_status,
        "Molecular_Weight": mw,
        "LogP": logp,
        "TPSA": tpsa,
        "HBD": hbd,
        "HBA": hba
    }

st.title("🧪 In Silico Chemical Toxicity & ADMET Prediction Platform")
st.write("Predict chemical toxicity endpoints, drug-likeness (Lipinski's Rule of Five), and molecular descriptors in real time.")

tab1, tab2 = st.tabs(["🔬 Single Molecule Predictor", "📂 Batch Screening (CSV / Multi-SMILES)"])

with tab1:
    examples = {
        "Aspirin (Low Risk)": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Ibuprofen (Low Risk)": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Acetaminophen / Paracetamol (Low Risk)": "CC(=O)NC1=CC=C(C=C1)O",
        "Caffeine (Low Risk)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Glucose (Low Risk)": "C(C1C(C(C(C(O1)O)O)O)O)O",
        "Ethanol (Low Risk)": "CCO",
        "Acetic Acid (Low Risk)": "CC(=O)O",
        "Benzoic Acid (Low Risk)": "C1=CC=C(C=C1)C(=O)O",
        "Salbutamol (Low Risk)": "CC(C)(C)NCC(O)C1=CC(CO)=C(O)C=C1",
        "Nicotine (High Risk)": "CN1CCCC1C2=CN=CC=C2",
        "Nitrobenzene (High Risk)": "C1=CC=C(C=C1)[N+](=O)[O-]",
        "Phenanthrene - PAH (High Risk)": "C1=CC=C2C(=C1)C=CC3=CC=CC=C32",
        "Pyrene - PAH (High Risk)": "C1=CC2=C3C(=C1)C=CC4=CC=CC(=C43)C=C2",
        "DDT Pesticide (High Risk)": "ClC1=CC=C(C=C1)C(C2=CC=C(Cl)C=C2)C(Cl)(Cl)Cl",
        "Azobenzene (High Risk)": "C1=CC=C(C=C1)N=NC2=CC=CC=C2",
        "Chlorobenzene (High Risk)": "C1=CC=C(C=C1)Cl",
        "Quinoline (High Risk)": "C1=CC=C2C(=C1)C=CC=N2",
        "4-Acetylaminobiphenyl - Carcinogen (High Risk)": "CC(=O)NC1=CC=C(C=C1)C2=CC=CC=C2",
        "p-Phenylenediamine (High Risk)": "C1=CC(=CC=C1N)N"
    }

    col1, col2 = st.columns(2)

    with col1:
        selected_example = st.selectbox("Quick Example Molecules:", list(examples.keys()))
        smiles_input = st.text_input("Or Enter Custom SMILES String:", value=examples[selected_example])
        st.button("🔬 Run Toxicity Prediction", type="primary")

    res = analyze_molecule(smiles_input)

    if res is None:
        st.error("❌ Invalid SMILES string. Please check the chemical notation.")
    else:
        risk_percent = res["Toxicity_Probability_%"]
        with col2:
            if risk_percent >= 65:
                st.error(f"### 🔴 HIGH TOXICITY RISK ({risk_percent}%)")
            elif risk_percent >= 35:
                st.warning(f"### 🟡 MODERATE TOXICITY RISK ({risk_percent}%)")
            else:
                st.success(f"### 🟢 LOW TOXICITY RISK ({risk_percent}%)")

            img = Draw.MolToImage(res["mol"], size=(300, 300))
            st.image(img, caption="2D Chemical Structure")

        st.divider()
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Toxicity Endpoints")
            st.write(f"- **QSAR Probability:** `{risk_percent}%`")
            st.write(f"- **Hepatotoxicity:** {res['Hepatotoxicity']}")
            st.write(f"- **Ames Mutagenicity:** {res['Ames_Mutagenicity']}")
            st.write(f"- **hERG Cardiotoxicity:** {res['hERG_Cardiotoxicity']}")

        with c2:
            st.subheader("Physicochemical Properties")
            st.write(f"- **Molecular Weight:** `{res['Molecular_Weight']} g/mol`")
            st.write(f"- **LogP (Lipophilicity):** `{res['LogP']}`")
            st.write(f"- **TPSA:** `{res['TPSA']} Å²`")
            st.write(f"- **H-Bond Donors / Acceptors:** `{res['HBD']}` / `{res['HBA']}`")

        with c3:
            st.subheader("Drug-Likeness (Lipinski)")
            st.write(f"- **Status:** **{res['Lipinski_Rule_of_5']}**")
            st.write(f"- MW ≤ 500: {'✅' if res['Molecular_Weight'] <= 500 else '❌'}")
            st.write(f"- LogP ≤ 5: {'✅' if res['LogP'] <= 5 else '❌'}")
            st.write(f"- HBD ≤ 5 & HBA ≤ 10: {'✅' if (res['HBD'] <= 5 and res['HBA'] <= 10) else '❌'}")

        report_df = pd.DataFrame([{k: v for k, v in res.items() if k != "mol"}])
        st.download_button(
            label="📥 Download Prediction Report (CSV)",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="toxicity_prediction_report.csv",
            mime="text/csv"
        )

with tab2:
    st.subheader("Screen Multiple Molecules at Once")
    st.write("Upload a CSV file containing a **`SMILES`** column, or paste multiple SMILES below (one per line).")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    smiles_list_input = st.text_area(
        "Or Paste SMILES List (one per line):",
        value="CC(=O)OC1=CC=CC=C1C(=O)O\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C\nC1=CC=C(C=C1)[N+](=O)[O-]\nCN1CCCC1C2=CN=CC=C2",
        height=130
    )

    if st.button("🚀 Run Batch Prediction", type="primary"):
        smiles_to_run = []
        if uploaded_file is not None:
            df_in = pd.read_csv(uploaded_file)
            if "SMILES" in df_in.columns:
                smiles_to_run = df_in["SMILES"].dropna().tolist()
            else:
                st.error("Your CSV file must have a column named 'SMILES'.")
        else:
            smiles_to_run = [line.strip() for line in smiles_list_input.splitlines() if line.strip()]

        if smiles_to_run:
            results = []
            for s in smiles_to_run:
                r = analyze_molecule(s)
                if r:
                    results.append({k: v for k, v in r.items() if k != "mol"})
                else:
                    results.append({"SMILES": s, "Risk_Level": "Invalid SMILES"})

            out_df = pd.DataFrame(results)
            st.dataframe(out_df, use_container_width=True)
            st.download_button(
                label="📥 Download Full Batch Results (CSV)",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="batch_toxicity_predictions.csv",
                mime="text/csv"
            )
