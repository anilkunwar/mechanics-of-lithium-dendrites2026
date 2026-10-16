import streamlit as st
import pyvista as pv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import glob
import io

# ==========================================
# Helper Functions
# ==========================================
def get_sequence_number(path):
    match = re.search(r"-s(\d+)$", path)
    if match:
        return int(match.group(1))
    return 0

def get_valid_blocks(obj):
    blocks = []
    if obj is None:
        return blocks
    if isinstance(obj, pv.MultiBlock):
        for block in obj:
            blocks.extend(get_valid_blocks(block))
    else:
        if hasattr(obj, "n_cells") and obj.n_cells > 0:
            blocks.append(obj)
    return blocks

def calculate_continuous_area(mesh, eta_name, electrode_end, eta_low=0.0, eta_high=1.0):
    if eta_name not in mesh.point_data:
        raise KeyError(f"no {eta_name}. now: {list(mesh.point_data.keys())}")
    if eta_high <= eta_low:
        raise ValueError("eta_high larger than eta_low")

    quad_ids = np.flatnonzero(mesh.celltypes == 9)
    if quad_ids.size == 0:
        raise ValueError(f"no QUAD4; now: {np.unique(mesh.celltypes)}")
    
    mesh = mesh.extract_cells(quad_ids)
    node_ids = mesh.cells.reshape(-1, 5)[:, 1:]

    x_nodes = mesh.points[node_ids, 0]
    y_nodes = mesh.points[node_ids, 1]
    eta_nodes = np.asarray(mesh.point_data[eta_name], dtype=float)[node_ids]

    gauss_points, gauss_weights = np.polynomial.legendre.leggauss(3)
    dendrite_area = 0.0
    electrolyte_area = 0.0

    for i, r in enumerate(gauss_points):
        for j, s in enumerate(gauss_points):
            N = 0.25 * np.array([(1-r)*(1-s), (1+r)*(1-s), (1+r)*(1+s), (1-r)*(1+s)])
            dN_dr = 0.25 * np.array([-(1-s), (1-s), (1+s), -(1+s)])
            dN_ds = 0.25 * np.array([-(1-r), -(1+r), (1+r), (1-r)])

            x_q = x_nodes @ N
            eta_q = eta_nodes @ N
            q = np.clip((eta_q - eta_low) / (eta_high - eta_low), 0.01, 0.99)
            h_q = q**3 * (6*q**2 - 15*q + 10)

            dx_dr, dx_ds = x_nodes @ dN_dr, x_nodes @ dN_ds
            dy_dr, dy_ds = y_nodes @ dN_dr, y_nodes @ dN_ds
            jacobian = np.abs(dx_dr * dy_ds - dx_ds * dy_dr)
            dA = gauss_weights[i] * gauss_weights[j] * jacobian

            region = x_q > electrode_end
            dendrite_area += np.sum(h_q[region] * dA[region])
            electrolyte_area += np.sum(dA[region])

    return dendrite_area, electrolyte_area

# ==========================================
# Streamlit App UI
# ==========================================
st.set_page_config(page_title="Dendrite Area Calculator", layout="wide")
st.title("🌲 Dendrite Continuous Area Calculator")

st.sidebar.header("⚙️ Configuration")

# 1. Robust Relative Path Resolution
# This automatically finds the correct /mount/src/... path on Streamlit Cloud
script_dir = os.path.dirname(os.path.abspath(__file__))
default_folder = os.path.join(script_dir, "test_dataset")

folder = st.sidebar.text_input("Folder Path", default_folder)
normalized_path = os.path.normpath(folder)

base_file = ""
base_file_names = []

# 2. Robust Directory Checking (Using your suggested approach)
if not os.path.exists(normalized_path):
    st.sidebar.error(f"❌ Directory not found: {normalized_path}")
    st.sidebar.info("💡 **Streamlit Cloud Tip:** Ensure `test_dataset` is committed to GitHub and NOT in your `.gitignore`.")
elif not os.path.isdir(normalized_path):
    st.sidebar.warning(f"⚠️ Path '{normalized_path}' exists but is not a directory.")
else:
    discovered_bases = set()
    try:
        # List all files in the folder
        for filename in os.listdir(normalized_path):
            # Match base.e and base.e-s0001
            match = re.match(r"^(.*\.e)(?:-s\d+)?$", filename)
            if match:
                file_full_path = os.path.join(normalized_path, filename)
                if os.path.isfile(file_full_path):
                    discovered_bases.add(match.group(1))
    except PermissionError:
        st.sidebar.warning(f"⚠️ Permission denied reading folder: {normalized_path}")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error scanning folder {normalized_path}: {e}")

    base_file_names = sorted(list(discovered_bases))

    if not base_file_names:
        st.sidebar.info(f"📁 Folder '{normalized_path}' contains no .e files.")
        base_file = st.sidebar.text_input("Enter Base File Name manually", "solid_change.e")
    else:
        st.sidebar.success(f"📁 Found {len(base_file_names)} base file(s).")
        default_index = 0
        if "solid_change.e" in base_file_names:
            default_index = base_file_names.index("solid_change.e")
            
        base_file = st.sidebar.selectbox(
            "Detected Base .e Files",
            base_file_names,
            index=default_index,
        )

eta_name = st.sidebar.text_input("Phase-field Variable Name", "eta")
electrode_end = st.sidebar.number_input("Electrode End (x ≤ value)", value=5.0, format="%.2f")

# ==========================================
# Processing Logic
# ==========================================
if st.sidebar.button("🚀 Run Calculation"):
    if not base_file:
        st.error("Please select or enter a base file name.")
        st.stop()

    filename = os.path.join(normalized_path, base_file)

    # Find sequence files
    sequence_files = glob.glob(filename + "-s*")
    sequence_files = [
        f for f in sequence_files
        if re.fullmatch(re.escape(os.path.basename(filename)) + r"-s\d+", os.path.basename(f))
    ]
    sequence_files.sort(key=get_sequence_number)

    if os.path.isfile(filename):
        all_files = [filename] + sequence_files
    elif sequence_files:
        all_files = sequence_files
    else:
        st.error(f"No base file or sequence files found for: {filename}")
        st.stop()

    st.write(f"Found **{len(all_files)}** file(s) to process.")
    
    results = []
    total_files = len(all_files)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for file_index, current_file in enumerate(all_files):
        status_text.text(f"⏳ Processing {file_index + 1}/{total_files}: {os.path.basename(current_file)}")
        
        try:
            reader = pv.get_reader(current_file, force_ext=".e")
        except Exception as e:
            st.warning(f"⚠️ Could not read {os.path.basename(current_file)}: {e}")
            progress_bar.progress((file_index + 1) / total_files)
            continue
        
        for time in reader.time_values:
            reader.set_active_time_value(time)
            raw_mesh = reader.read()
            
            if isinstance(raw_mesh, pv.MultiBlock):
                blocks = get_valid_blocks(raw_mesh)
                if len(blocks) == 0: continue
                mesh = blocks[0] if len(blocks) == 1 else pv.MultiBlock(blocks).combine()
            else:
                mesh = raw_mesh
            
            try:
                dendrite_area, electrolyte_area = calculate_continuous_area(
                    mesh=mesh, eta_name=eta_name, electrode_end=electrode_end
                )
                area_fraction = (dendrite_area / electrolyte_area * 100.0) if electrolyte_area > 0 else np.nan
                results.append([file_index, os.path.basename(current_file), time, dendrite_area, area_fraction])
            except Exception as e:
                st.warning(f"⚠️ Error at time {time}: {e}")
        
        progress_bar.progress((file_index + 1) / total_files)
        
    status_text.text("✅ Finished!")
    
    if results:
        df = pd.DataFrame(results, columns=["File_Index", "File", "Time", "Dendrite_Area", "Area_Fraction_percent"])
        st.subheader("📊 Results Data")
        st.dataframe(df)
        
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
            
        st.download_button("📥 Download Excel File", excel_buffer.getvalue(), "dendrite_area_results.xlsx")
        
        st.subheader("📈 Dendrite Area Fraction vs Time")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["Time"], df["Area_Fraction_percent"], marker="o", markersize=2, linewidth=1.5)
        ax.set_xlabel("Time"); ax.set_ylabel("Dendrite Area Fraction (%)"); ax.grid(True)
        st.pyplot(fig)
    else:
        st.warning("⚠️ No results were generated.")
