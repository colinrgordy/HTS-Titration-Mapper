import streamlit as st
import pandas as pd
import json
import os
import io
import tempfile
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import rdMolDraw2D

st.set_page_config(page_title="HTS Titration & Echo Mapper", page_icon="🧪", layout="wide")

st.title("NCATS HTS Titration & Echo Mapping Engine")
st.markdown("Designed for mapping compound titrations on 384-well source plates and generating transposed, color-coded 1536-well destination maps for Echo acoustic liquid handling.")

# --- Distinct Color Palette for the 12 Compounds + 1 DMSO ---
COLORS = [
    "#1abc9c",  # Teal (Compound 1)
    "#3498db",  # Blue (Compound 2)
    "#9b59b6",  # Purple (Compound 3)
    "#e74c3c",  # Red (Compound 4)
    "#e67e22",  # Orange (Compound 5)
    "#f1c40f",  # Yellow (Compound 6)
    "#2ecc71",  # Green (Compound 7)
    "#e84393",  # Pink (Compound 8)
    "#6c5ce7",  # Indigo (Compound 9)
    "#00cec9",  # Cyan (Compound 10)
    "#ff7675",  # Coral (Compound 11)
    "#b2bec3"   # Slate Gray (Compound 12)
]
DMSO_COLOR = "#64748b"  # Cool slate gray for DMSO Block

# --- Helper Function for Clean Concentration Formatting ---
def format_conc(val_in_mm):
    if val_in_mm is None:
        return "N/A"
    if val_in_mm == 0.0:
        return "0.00 mM (DMSO)"
    
    if val_in_mm >= 1.0:
        return f"{val_in_mm:.2f} mM"
    elif val_in_mm >= 0.001:
        return f"{val_in_mm * 1000:.2f} µM"
    elif val_in_mm >= 0.000001:
        return f"{val_in_mm * 1000000:.2f} nM"
    else:
        return f"{val_in_mm * 1000000000:.2f} pM"

# --- Sidebar Inputs ---
st.sidebar.header("1. Experiment & Plate Names")
exp_name = st.sidebar.text_input("Experiment / Assay Name", value="Prion_Screen_Validation")
source_plate_name = st.sidebar.text_input("Source Plate Name (384)", value="Source[1]")
dest_plate_name = st.sidebar.text_input("Destination Plate Name (1536)", value="Destination[1]")

# Sanitize experiment name for file outputs
clean_exp_name = "".join([c if c.isalnum() or c in ['_', '-'] else '_' for c in exp_name]).strip().lower()
if not clean_exp_name:
    clean_exp_name = "hts_titration"

st.sidebar.header("2. Assay Setup")
start_conc = st.sidebar.number_input("Starting Concentration (mM)", value=10.0, step=1.0)
dil_factor = st.sidebar.number_input("Dilution Factor (e.g., 3 for 1:3)", value=3.0, step=1.0)

st.sidebar.header("3. Volume & Dilution Math")
echo_vol_nl = st.sidebar.number_input("Echo Transfer Volume (nL)", value=20.0, step=2.5)
assay_vol_ul = st.sidebar.number_input("Assay Buffer Volume (µL)", value=5.0, step=0.5)

total_vol_nl = (assay_vol_ul * 1000) + echo_vol_nl
echo_dilution_factor = total_vol_nl / echo_vol_nl

st.sidebar.metric("Echo Dilution Factor", f"1 : {echo_dilution_factor:.1f}")

st.sidebar.header("4. Plate Layout Settings")
num_replicates = st.sidebar.slider("Number of Replicates (on 1536 Plate)", min_value=1, max_value=4, value=3, step=1)
replicates = st.sidebar.checkbox("Generate Technical Duplicates on Source?", value=False, 
                                  help="If checked, duplicates titration in Cols 13-23 on the 384 plate.")

st.sidebar.header("5. Destination 1536 Positioning")
row_labels_1536 = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF']
start_row_1536 = st.sidebar.selectbox("Starting Row (1536)", row_labels_1536, index=0)
start_col_1536 = st.sidebar.number_input("Starting Column (1536)", min_value=1, max_value=48, value=1, step=1)

# --- Default Paste Data ---
default_paste = """NCGC00091454,CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5
NCGC00123456,CCOc1ccc2nc(sc2c1)S(=O)(=O)N
NCGC00123457,CC(=O)Nc1ccc(cc1)O
NCGC00123458,CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13
NCGC00123459,CC12CCC3C(C1CCC2=O)CCC4=CC(=O)CCC34C
NCGC00123460,Cc1onc(c1C(=O)Nc2ccc(cc2)C(F)(F)F)C
NCGC00123461,CC(=O)O
NCGC00123462,CN(C)C(=N)N=C(N)N
NCGC00123463,Clc1ccc(cc1)C(c2ccccc2)N3CCN(CC3)CCOCC
NCGC00123464,CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccccc3)C(=O)N2[C@H]1C(=O)O
NCGC00123465,CN1c2ccccc2C(=O)N(C)C1=O
NCGC00123466,CC(=O)Oc1ccccc1C(=O)O"""

# --- Dual Input Interface ---
st.markdown("### Load Compound Library")
input_method = st.radio("Choose Input Method", ["Upload SDF File", "Paste IDs & SMILES"], index=1, horizontal=True)

compounds = []

if input_method == "Paste IDs & SMILES":
    raw_input = st.text_area("Paste NCGC IDs and SMILES (Format: ID,SMILES - one per line)", value=default_paste, height=200)
    for line in raw_input.strip().split("\n"):
        if "," in line:
            parts = line.split(",")
            compounds.append({"ID": parts[0].strip(), "SMILES": parts[1].strip()})
else:
    uploaded_file = st.file_uploader("Choose an SDF file", type=["sdf"])
    if uploaded_file is not None:
        with st.spinner("Parsing chemical structures from SDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            
            try:
                supplier = Chem.SDMolSupplier(temp_path)
                for idx, mol in enumerate(supplier):
                    if mol is None: continue
                    
                    sample_id = None
                    for prop_name in ['SAMPLE_ID', 'Name', 'ID', 'sample_id', 'id', 'NCGC_ID', 'NCGC_id']:
                        if mol.HasProp(prop_name):
                            sample_id = mol.GetProp(prop_name)
                            break
                    
                    if not sample_id: 
                        sample_id = f"UNKNOWN_{idx}"
                    else:
                        if '-' in str(sample_id):
                            sample_id = str(sample_id).split('-')[0]
                    
                    try:
                        smiles = Chem.MolToSmiles(mol)
                        if smiles and smiles.strip() != "":
                            compounds.append({"ID": sample_id, "SMILES": smiles})
                    except:
                        continue
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

if len(compounds) > 12:
    st.warning(f"⚠️ {len(compounds)} compounds detected. Only the first 12 will be processed to fit the Row A-L layout.")
    compounds = compounds[:12]

# --- Boundary Validation Check ---
boundary_error = False
if compounds:
    start_row_idx = row_labels_1536.index(start_row_1536)
    total_rows_needed = 11  # exactly 11 titration points
    max_row_idx_needed = start_row_idx + total_rows_needed
    
    total_cols_needed = 13 * num_replicates
    max_col_needed = start_col_1536 + total_cols_needed - 1

    if max_row_idx_needed > 32 or max_col_needed > 48:
        boundary_error = True
        st.error(f"⚠️ **Layout Boundary Error:** The requested layout exceeds the physical limits of the 1536-well plate!\n\n"
                 f"* **Rows Needed:** {max_row_idx_needed} / 32 (Row '{row_labels_1536[max_row_idx_needed - 1]}' is the last row needed)\n"
                 f"* **Columns Needed:** {max_col_needed} / 48\n\n"
                 f"Please adjust your starting row/column offsets or reduce the number of replicates.")

# --- Calculation & Mapping Engine ---
if compounds and not boundary_error:
    # Generate Source Concentrations (in mM)
    source_concentrations = []
    current_c = start_conc
    for i in range(11):
        source_concentrations.append(current_c)
        current_c = current_c / dil_factor

    source_rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    source_records = []
    echo_records = []

    cols_to_map = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    if replicates:
        cols_to_map += [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    # --- PART A: Map Compound Rows A-L ---
    for comp_idx, comp in enumerate(compounds):
        row_letter = source_rows[comp_idx]
        comp_color = COLORS[comp_idx]

        # Generate Structure SVGs[span_0](start_span)[span_0](end_span)
        svg_text = ""
        try:
            mol = Chem.MolFromSmiles(comp["SMILES"])
            if mol:
                drawer = rdMolDraw2D.MolDraw2DSVG(160, 160)
                clean_mol = rdMolDraw2D.PrepareMolForDrawing(mol)
                drawer.DrawMolecule(clean_mol)
                drawer.FinishDrawing()
                svg_text = drawer.GetDrawingText()
        except:
            svg_text = ""

        for col in range(1, 25):
            well_384 = f"{row_letter}{col:02d}"
            is_titration = col in cols_to_map
            
            if is_titration:
                if col <= 11:
                    point_idx = col - 1
                    base_offset = 0
                else:
                    point_idx = col - 13
                    base_offset = 11 * num_replicates
                
                src_conc = source_concentrations[point_idx]
                dest_conc = src_conc / echo_dilution_factor
                label = f"Point {point_idx + 1}"
                
                source_records.append({
                    "Source_Well": well_384,
                    "NCGC_ID": comp["ID"],
                    "SMILES": comp["SMILES"],
                    "Source_Conc_mM": src_conc,
                    "Dest_Conc_mM": dest_conc,
                    "Type": label,
                    "SVG": svg_text,
                    "Color": comp_color
                })

                # --- Map to 1536-well plate (Vertical Transposition) ---
                dest_row_letter = row_labels_1536[start_row_idx + point_idx]
                
                for rep in range(1, num_replicates + 1):
                    dest_col = start_col_1536 + (comp_idx * num_replicates) + (rep - 1)
                    well_1536 = f"{dest_row_letter}{dest_col:02d}"
                    
                    echo_records.append({
                        "Source Plate Name": source_plate_name, # changed[span_1](start_span)[span_1](end_span)
                        "Source Well": well_384,
                        "Destination Plate Name": dest_plate_name, # changed[span_2](start_span)[span_2](end_span)
                        "Destination Well": well_1536,
                        "Transfer Volume (nL)": echo_vol_nl,
                        "NCGC_ID": comp["ID"],
                        "SMILES": comp["SMILES"],
                        "Source_Conc": format_conc(src_conc),
                        "Assay_Conc": format_conc(dest_conc),
                        "Type": f"{label} (Rep {rep})",
                        "Color": comp_color
                    })
            else:
                source_records.append({
                    "Source_Well": well_384,
                    "NCGC_ID": "Empty",
                    "SMILES": "",
                    "Source_Conc_mM": None,
                    "Dest_Conc_mM": None,
                    "Type": "Empty",
                    "SVG": "",
                    "Color": "#f1f5f9"
                })

    # --- PART B: Map Dedicated Row M for DMSO ---
    dmso_row_letter = "M"

    for col in range(1, 25):
        well_384 = f"{dmso_row_letter}{col:02d}"
        is_dmso_well = col in cols_to_map
        
        if is_dmso_well:
            if col <= 11:
                point_idx = col - 1
                base_offset = 0
            else:
                point_idx = col - 13
                base_offset = 11 * num_replicates

            source_records.append({
                "Source_Well": well_384,
                "NCGC_ID": "DMSO",
                "SMILES": "",
                "Source_Conc_mM": 0.0,
                "Dest_Conc_mM": 0.0,
                "Type": "DMSO Control",
                "SVG": "",
                "Color": DMSO_COLOR
            })

            dest_row_letter = row_labels_1536[start_row_idx + point_idx]
            for rep in range(1, num_replicates + 1):
                dest_col = start_col_1536 + (12 * num_replicates) + (rep - 1)
                well_1536 = f"{dest_row_letter}{dest_col:02d}"

                echo_records.append({
                    "Source Plate Name": source_plate_name, # changed[span_3](start_span)[span_3](end_span)
                    "Source Well": well_384,
                    "Destination Plate Name": dest_plate_name, # changed[span_4](start_span)[span_4](end_span)
                    "Destination Well": well_1536,
                    "Transfer Volume (nL)": echo_vol_nl,
                    "NCGC_ID": "DMSO",
                    "SMILES": "",
                    "Source_Conc": format_conc(0.0),
                    "Assay_Conc": format_conc(0.0),
                    "Type": f"DMSO Control (Rep {rep})",
                    "Color": DMSO_COLOR
                })
        else:
            source_records.append({
                "Source_Well": well_384,
                "NCGC_ID": "Empty",
                "SMILES": "",
                "Source_Conc_mM": None,
                "Dest_Conc_mM": None,
                "Type": "Empty",
                "SVG": "",
                "Color": "#f1f5f9"
            })

    # Fill remaining rows (N, O, P) in 384 plate as empty
    for row_letter in ['N', 'O', 'P']:
        for col in range(1, 25):
            source_records.append({
                "Source_Well": f"{row_letter}{col:02d}",
                "NCGC_ID": "Empty",
                "SMILES": "",
                "Source_Conc_mM": None,
                "Dest_Conc_mM": None,
                "Type": "Empty",
                "SVG": "",
                "Color": "#f1f5f9"
            })

    df_source = pd.DataFrame(source_records)
    df_echo = pd.DataFrame(echo_records)

    # --- Metrics Section ---
    st.success("Plates compiled and dilution series calculated!")
    col1, col2, col3 = st.columns(3)
    col1.metric("Compounds Mapped", len(compounds))
    col2.metric("Point 11 Source Conc", format_conc(source_concentrations[10]))
    col3.metric("Point 11 Final Assay Conc", format_conc(source_concentrations[10] / echo_dilution_factor))

    # --- UI Downloads ---
    st.markdown("### Download Plate Layout Files")
    down_col1, down_col2 = st.columns(2)
    
    # 📝 EXCEL GENERATOR (With two dedicated worksheets!)
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        # Sheet 1: Echo transfer instructions
        df_echo_excel = df_echo.drop(columns=['Color'], errors='ignore')
        df_echo_excel.to_excel(writer, index=False, sheet_name="Echo_Transfer_Manifest")
        
        # Sheet 2: Clean 384 source plate bench log
        df_source_excel = df_source.copy()
        df_source_excel["Source_Conc_Formatted"] = df_source_excel["Source_Conc_mM"].apply(format_conc)
        df_source_excel["Assay_Conc_Formatted"] = df_source_excel["Dest_Conc_mM"].apply(format_conc)
        df_source_excel = df_source_excel.drop(columns=['Source_Conc_mM', 'Dest_Conc_mM', 'SVG', 'Color'], errors='ignore')
        df_source_excel.to_excel(writer, index=False, sheet_name="Source_384_Plate_Map")
        
    excel_buffer.seek(0)

    down_col1.download_button(
        label="Download Complete Layout Workbook (Excel)",
        data=excel_buffer,
        file_name=f"{clean_exp_name}_layout_workbook.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    # --- JSON payload construction for HTML template ---
    source_json_dict = {}
    for _, r in df_source.iterrows():
        source_json_dict[r["Source_Well"]] = {
            "id": r["NCGC_ID"],
            "smiles": r["SMILES"],
            "source_conc": format_conc(r["Source_Conc_mM"]),
            "dest_conc": format_conc(r["Dest_Conc_mM"]),
            "type": r["Type"],
            "img": r["SVG"],
            "color": r["Color"]
        }

    dest_json_dict = {}
    for _, r in df_echo.iterrows():
        dest_json_dict[r["Destination Well"]] = {
            "id": r["NCGC_ID"],
            "smiles": r["SMILES"],
            "source_conc": r["Source_Conc"],
            "dest_conc": r["Assay_Conc"],
            "type": r["Type"],
            "source_well": r["Source Well"],
            "color": r["Color"]
        }

    # Generate HTML code with dynamic JS color rendering, custom title, and legend generation
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{exp_name} - Interactive Plate Map</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f8f9fa; margin: 0; padding: 20px; color: #333; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e9ecef; padding-bottom: 15px; margin-bottom: 20px; }}
        h1 {{ margin: 0; font-size: 24px; color: #1e293b; }}
        .toggle-btn {{ padding: 8px 16px; font-size: 14px; border-radius: 6px; border: 1px solid #cbd5e1; outline: none; background: white; cursor: pointer; font-weight: 600; transition: all 0.2s; }}
        .toggle-btn.active {{ background: #2563eb; color: white; border-color: #2563eb; }}
        .main-container {{ display: flex; gap: 24px; align-items: flex-start; }}
        .plate-box {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; overflow-x: auto; }}
        
        .map-legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; font-size: 11px; font-weight: 600; justify-content: center; background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; max-width: 800px; margin-left: auto; margin-right: auto; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; padding: 4px 8px; background: white; border-radius: 4px; border: 1px solid #e2e8f0; }}
        
        .grid-384 {{ display: grid; grid-template-columns: 30px repeat(24, 24px); gap: 4px; align-items: center; justify-items: center; }}
        .well-384 {{ width: 20px; height: 20px; border-radius: 50%; background-color: #f1f5f9; border: 1px solid #cbd5e1; cursor: pointer; transition: all 0.1s ease; }}
        
        .grid-1536 {{ display: grid; grid-template-columns: 30px repeat(48, 14px); gap: 2px; align-items: center; justify-items: center; }}
        .well-1536 {{ width: 11px; height: 11px; border-radius: 50%; background-color: #f1f5f9; border: 1px solid #e2e8f0; cursor: pointer; transition: all 0.1s ease; }}
        
        .col-header {{ font-size: 9px; font-weight: bold; color: #64748b; text-align: center; }}
        .row-header {{ font-size: 9px; font-weight: bold; color: #64748b; text-align: center; display: flex; align-items: center; justify-content: center; }}
        
        .well:hover {{ transform: scale(1.3); border-color: #1e293b !important; box-shadow: 0 0 4px rgba(0,0,0,0.3); z-index: 10; }}
        .well.active-well {{ border-color: #1e3a8a !important; box-shadow: 0 0 0 2px #2563eb !important; }}
        
        .well.dmso-well {{ border: 2px dashed #475569 !important; position: relative; }}
        .well.dmso-well::after {{ content: 'D'; color: rgba(255,255,255,0.8); font-size: 8px; font-weight: bold; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); pointer-events: none; }}
        .well-1536.dmso-well::after {{ font-size: 6px; }}

        .display-panel {{ flex: 1; min-width: 400px; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; max-height: 85vh; overflow-y: auto; }}
        .panel-title {{ font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #1e293b; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }}
        .compound-card {{ display: flex; flex-direction: column; gap: 15px; padding: 16px; border: 1px solid #e2e8f0; border-radius: 8px; background: #f8fafc; }}
        .compound-id {{ font-size: 18px; font-weight: bold; color: #2563eb; margin-bottom: 4px; }}
        .struct-img {{ width: 200px; height: 200px; background: white; border: 1px solid #e2e8f0; border-radius: 6px; display: flex; align-items: center; justify-content: center; overflow: hidden; align-self: center; }}
        .struct-img img, .struct-img svg {{ max-width: 100%; max-height: 100%; }}
        .placeholder-text {{ color: #94a3b8; font-style: italic; text-align: center; margin-top: 50px; }}
    </style>
</head>
<body>

    <div class="header">
        <h1>{exp_name} - Plate Navigator</h1>
        <div>
            <button id="btn384" class="toggle-btn active" onclick="setView('384')">{source_plate_name} (384)</button>
            <button id="btn1536" class="toggle-btn" onclick="setView('1536')">{dest_plate_name} (1536)</button>
        </div>
    </div>

    <div class="main-container">
        <div class="plate-box">
            <div id="legendContainer" class="map-legend"></div>
            <div id="gridContainer"></div>
        </div>
        <div class="display-panel">
            <div id="panelTitle" class="panel-title">Well Inspector</div>
            <div id="inspectorContent">
                <div class="placeholder-text">Click any populated well on the left to inspect its layout mapping, concentration values, and chemical structures...</div>
            </div>
        </div>
    </div>

    <script>
        const sourceDb = {json.dumps(source_json_dict)};
        const destDb = {json.dumps(dest_json_dict)};
        
        let currentView = '384';
        
        const rows384 = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P'];
        const rows1536 = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF'];

        function renderLegend() {{
            const legend = document.getElementById('legendContainer');
            legend.innerHTML = '';
            
            const seen = new Set();
            const list = [];
            
            Object.values(sourceDb).forEach(val => {{
                if (val.id !== 'Empty' && val.id !== 'DMSO' && !seen.has(val.id)) {{
                    seen.add(val.id);
                    list.push({{ id: val.id, color: val.color }});
                }}
            }});
            
            list.forEach(item => {{
                let pill = document.createElement('div');
                pill.className = 'legend-item';
                pill.innerHTML = `
                    <div style="width: 12px; height: 12px; border-radius: 50%; background-color: ${{item.color}}; border: 1px solid rgba(0,0,0,0.15);"></div>
                    <span>${{item.id}}</span>
                `;
                legend.appendChild(pill);
            }});
            
            let dmsoPill = document.createElement('div');
            dmsoPill.className = 'legend-item';
            dmsoPill.innerHTML = `
                <div style="width: 12px; height: 12px; border-radius: 50%; background-color: ${DMSO_COLOR}; border: 2px dashed #475569;"></div>
                <span>DMSO Vehicle Control</span>
            `;
            legend.appendChild(dmsoPill);
        }}

        function setView(view) {{
            currentView = view;
            document.getElementById('btn384').classList.toggle('active', view === '384');
            document.getElementById('btn1536').classList.toggle('active', view === '1536');
            renderPlate();
        }}

        function renderPlate() {{
            const container = document.getElementById('gridContainer');
            container.innerHTML = '';
            
            if (currentView === '384') {{
                container.className = 'grid-384';
                container.appendChild(document.createElement('div'));
                for(let c=1; c<=24; c++) {{
                    let header = document.createElement('div');
                    header.className = 'col-header';
                    header.innerHTML = c;
                    container.appendChild(header);
                }}
                
                rows384.forEach(r => {{
                    let rHeader = document.createElement('div');
                    rHeader.className = 'row-header';
                    rHeader.innerHTML = r;
                    container.appendChild(rHeader);
                    
                    for(let c=1; c<=24; c++) {{
                        let wellName = r + (c < 10 ? '0'+c : c);
                        let wellDiv = document.createElement('div');
                        wellDiv.className = 'well well-384';
                        wellDiv.id = wellName;
                        
                        const data = sourceDb[wellName];
                        if (data && data.id !== 'Empty') {{
                            wellDiv.classList.add('populated');
                            wellDiv.style.backgroundColor = data.color;
                            
                            if (data.id === 'DMSO') {{
                                wellDiv.classList.add('dmso-well');
                            }}
                            
                            wellDiv.onclick = () => selectSourceWell(wellName, data, wellDiv);
                        }}
                        container.appendChild(wellDiv);
                    }}
                }});
            }} else {{
                container.className = 'grid-1536';
                container.appendChild(document.createElement('div'));
                for(let c=1; c<=48; c++) {{
                    let header = document.createElement('div');
                    header.className = 'col-header';
                    header.innerHTML = c;
                    container.appendChild(header);
                }}
                
                rows1536.forEach(r => {{
                    let rHeader = document.createElement('div');
                    rHeader.className = 'row-header';
                    rHeader.innerHTML = r;
                    container.appendChild(rHeader);
                    
                    for(let c=1; c<=48; c++) {{
                        let wellName = r + (c < 10 ? '0'+c : c);
                        let wellDiv = document.createElement('div');
                        wellDiv.className = 'well well-1536';
                        wellDiv.id = wellName;
                        
                        const data = destDb[wellName];
                        if (data) {{
                            wellDiv.classList.add('populated');
                            wellDiv.style.backgroundColor = data.color;
                            
                            if (data.id === 'DMSO') {{
                                wellDiv.classList.add('dmso-well');
                            }}
                            
                            wellDiv.onclick = () => selectDestWell(wellName, data, wellDiv);
                        }}
                        container.appendChild(wellDiv);
                    }}
                }});
            }}
        }}

        function selectSourceWell(wellName, data, element) {{
            document.querySelectorAll('.well').forEach(w => w.classList.remove('active-well'));
            element.classList.add('active-well');
            
            document.getElementById('panelTitle').innerHTML = "Source Well: " + wellName;
            
            const content = document.getElementById('inspectorContent');
            content.innerHTML = `
                <div class="compound-card">
                    <div class="compound-id" style="color: ${{data.color === '#f1f5f9' ? '#2563eb' : data.color}}">${{data.id}}</div>
                    <div><strong>Well Type:</strong> ${{data.type}}</div>
                    <div><strong>Source Plate Concentration:</strong> ${{data.source_conc}}</div>
                    <div><strong>Projected Assay Concentration:</strong> ${{data.dest_conc}}</div>
                    <div class="struct-img">
                        ${{data.img ? data.img : '<span style="color:#cbd5e1;">No Structure Available</span>'}}
                    </div>
                    <div style="font-size:11px; word-break:break-all; color:#64748b;">
                        <strong>SMILES:</strong> ${{data.smiles}}
                    </div>
                </div>
            `;
        }}

        function selectDestWell(wellName, data, element) {{
            document.querySelectorAll('.well').forEach(w => w.classList.remove('active-well'));
            element.classList.add('active-well');
            
            document.getElementById('panelTitle').innerHTML = "Destination Well (1536): " + wellName;
            
            const srcData = sourceDb[data.source_well];
            const svgContent = srcData ? srcData.img : '';
            
            const content = document.getElementById('inspectorContent');
            content.innerHTML = `
                <div class="compound-card">
                    <div class="compound-id" style="color: ${{data.color}}">${{data.id}}</div>
                    <div><strong>Mapped From Source:</strong> ${{data.source_well}}</div>
                    <div><strong>Well Type:</strong> ${{data.type}}</div>
                    <div><strong>Source Plate Concentration:</strong> ${{data.source_conc}}</div>
                    <div><strong>Assay Concentration (in Well):</strong> ${{data.dest_conc}}</div>
                    <div class="struct-img">
                        ${{svgContent ? svgContent : '<span style="color:#cbd5e1;">No Structure Available</span>'}}
                    </div>
                    <div style="font-size:11px; word-break:break-all; color:#64748b;">
                        <strong>SMILES:</strong> ${{data.smiles}}
                    </div>
                </div>
            `;
        }}

        renderLegend();
        renderPlate();
    </script>
</body>
</html>
"""

    down_col2.download_button(
        label="Download Interactive HTML Plate Map (Visual Layout)",
        data=html_content,
        file_name=f"{clean_exp_name}_interactive_map.html",
        mime="text/html",
        use_container_width=True
    )

    # --- Interactive App Previews ---
    st.markdown("---")
    st.markdown("### 5. Interactive Previews")
    
    tabs = st.tabs([f"{source_plate_name} Map Preview", f"{dest_plate_name} Echo Mapped Manifest"])
    
    with tabs[0]:
        df_source_preview = df_source[df_source["NCGC_ID"] != "Empty"].copy()
        df_source_preview["Source_Conc_Formatted"] = df_source_preview["Source_Conc_mM"].apply(format_conc)
        df_source_preview["Assay_Conc_Formatted"] = df_source_preview["Dest_Conc_mM"].apply(format_conc)
        
        st.dataframe(
            df_source_preview[["Source_Well", "NCGC_ID", "Source_Conc_Formatted", "Assay_Conc_Formatted", "Type"]],
            use_container_width=True,
            hide_index=True
        )
        
    with tabs[1]:
        st.dataframe(
            df_echo[["Source Well", "Destination Well", "Transfer Volume (nL)", "NCGC_ID", "Source_Conc", "Assay_Conc", "Type"]],
            use_container_width=True,
            hide_index=True
        )
else:
    if not boundary_error:
        st.info("Please paste some structures or upload an SDF file above to generate your plate mapping profiles.")
