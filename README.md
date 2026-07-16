# HTS Titration & Echo Mapping Engine

An interactive web application designed to map compound titrations on 384-well source plates and generate transposed, color-coded 1536-well destination maps for Echo acoustic liquid handling.

## Key Features

* **Dynamic Dose-Response Calculations:** Automatically scales 11-point titration curves starting from a user-defined stock concentration (e.g., 10 mM) using customized dilution factors and exact Echo dilution math.
* **Vertical Transposition Mapping:** Transposes horizontal 384-well source series into vertical column blocks on high-density 1536-well plates, supporting customizable replicate counts (e.g., triplicates) and flexible starting offsets.
* **Interactive Visual Layouts:** Features high-fidelity, dual-view interactive HTML plate maps (384 source and 1536 destination) color-coded by compound with built-in hover tooltips and molecular structure previews.
* **Multisheet Excel Export:** Generates standardized, GLP-compliant `.xlsx` workbooks containing ready-to-upload Echo transfer manifests and formatted 384-well source bench logs for seamless ELN integration.
* **Chemical Structure Parsing:** Accepts manual text inputs (NCGC ID and SMILES) or direct `.sdf` file uploads, automatically parsing and rendering structures using RDKit.

## Technical Stack

* **Language:** Python 3.9
* **Framework:** Streamlit
* **Data Handling:** Pandas, OpenPyXL, & RDKit
* **Deployment:** Streamlit Community Cloud

## Background

Developed to eliminate manual calculation and transposition errors in drug-titration workflows. Specifically designed to automate the vertical column-block mapping of cherrypicked compounds against target proteins (such as prion proteins), this tool bridges the gap between manual benchtop serial dilutions on 384-well source plates and automated nanoliter-dispensing on 1536-well assay plates using the Echo acoustic liquid handler.

---

Made by Colin Gordy