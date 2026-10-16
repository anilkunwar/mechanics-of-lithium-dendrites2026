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

def calculate_continuous_area(
    mesh,
    eta_name,
    electrode_end,
    eta_low=0.0,
    eta_high=1.0
):
    if eta_name not in mesh.point_data:
        raise KeyError(
            f"no {eta_name}. now: {list(mesh.point_data.keys())}"
        )

    if eta_high <= eta_low:
        raise ValueError("eta_high larger than eta_low")

    # QUAD4
    quad_ids = np.flatnonzero(mesh.celltypes == 9)
    
    if quad_ids.size == 0:
        raise ValueError(
            f"no QUAD4; now: {np.unique(mesh.celltypes)}"
        )
    
    mesh = mesh.extract_cells(quad_ids)

    # QUAD4-4 point
    node_ids = mesh.cells.reshape(-1, 5)[:, 1:]

    x_nodes = mesh.points[node_ids, 0]
    y_nodes = mesh.points[node_ids, 1]
    eta_nodes = np.asarray(
        mesh.point_data[eta_name],
        dtype=float
    )[node_ids]

    gauss_points, gauss_weights = np.polynomial.legendre.leggauss(3)

    dendrite_area = 0.0
    electrolyte_area = 0.0

    for i, r in enumerate(gauss_points):
        for j, s in enumerate(gauss_points):

            # QUAD4 shape function
            N = 0.25 * np.array([
                (1-r)*(1-s),
                (1+r)*(1-s),
                (1+r)*(1+s),
                (1-r)*(1+s)
            ])

            dN_dr = 0.25 * np.array([
                -(1-s),
                 (1-s),
                 (1+s),
                -(1+s)
            ])

            dN_ds = 0.25 * np.array([
                -(1-r),
                -(1+r),
                 (1+r),
                 (1-r)
            ])

            x_q = x_nodes @ N
            eta_q = eta_nodes @ N

            q = np.clip(
                (eta_q - eta_low) / (eta_high - eta_low),
                0.01,
                0.99
            )

            # calculate h
            h_q = q**3 * (6*q**2 - 15*q + 10)

            # Jacobian
            dx_dr = x_nodes @ dN_dr
            dx_ds = x_nodes @ dN_ds
            dy_dr = y_nodes @ dN_dr
            dy_ds = y_nodes @ dN_ds

            jacobian = np.abs(
                dx_dr * dy_ds - dx_ds * dy_dr
            )

            dA = (
                gauss_weights[i]
                * gauss_weights[j]
                * jacobian
            )

            # IC
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

# 1. Default folder
default_folder = os.path.join(os.getcwd(), "test_dataset")
folder = st.sidebar.text_input("Folder Path", default_folder)

# 2. Discover base files and sequence files
base_file = ""
if os.path.isdir(folder):
    all_e_files = glob.glob(os.path.join(folder, "*.e*"))

    # Recognise both:
    # solid_change.e
    # solid_change.e-s0001, solid_change.e-s0002, ...
    discovered_bases = set()

    for path in all_e_files:
        name = os.path.basename(path)

        match = re.match(r"^(.*\.e)(?:-s\d+)?$", name)
        if match:
            discovered_bases.add(match.group(1))

    base_file_names = sorted(discovered_bases)

    if base_file_names:
        # Try to default to solid_change.e if it exists
        default_index = 0
        if "solid_change.e" in base_file_names:
            default_index = base_file_names.index("solid_change.e")
            
        base_file = st.sidebar.selectbox(
            "Detected Base .e Files",
            base_file_names,
            index=default_index,
        )
    else:
        st.sidebar.warning(
            "No .e files or .e-sNNNN sequence files detected."
        )
        base_file = st.sidebar.text_input(
            "Enter Base File Name manually",
            "solid_change.e",
        )
else:
    st.sidebar.error(f"Directory not found: {folder}")
    base_file = st.sidebar.text_input(
        "Enter Base File Name manually",
        "solid_change.e",
    )

eta_name = st.sidebar.text_input(
    "Phase-field Variable Name", "eta"
)
electrode_end = st.sidebar.number_input(
    "Electrode End (x ≤ value)",
    value=5.0,
    format="%.2f",
)

# ==========================================
# Processing Logic
# ==========================================
if st.sidebar.button("🚀 Run Calculation"):
    filename = os.path.join(folder, base_file)

    # Find the base file and all matching sequence files.
    sequence_files = glob.glob(filename + "-s*")
    sequence_files = [
        f for f in sequence_files
        if re.fullmatch(
            re.escape(os.path.basename(filename)) + r"-s\d+",
            os.path.basename(f),
        )
    ]
    sequence_files.sort(key=get_sequence_number)

    if os.path.isfile(filename):
        all_files = [filename] + sequence_files
    elif sequence_files:
        all_files = sequence_files
    else:
        st.error(
            f"No base file or sequence files found for: {filename}"
        )
        st.stop()

    st.write(f"Found **{len(all_files)}** file(s) to process.")
    st.write([os.path.basename(f) for f in all_files])

    results = []
    total_files = len(all_files)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Main Processing Loop
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
                if len(blocks) == 0:
                    continue
                elif len(blocks) == 1:
                    mesh = blocks[0]
                else:
                    mesh = pv.MultiBlock(blocks).combine()
            else:
                mesh = raw_mesh
            
            try:
                dendrite_area, electrolyte_area = calculate_continuous_area(
                    mesh=mesh,
                    eta_name=eta_name,
                    electrode_end=electrode_end,
                    eta_low=0.0,
                    eta_high=1.0
                )
                
                if electrolyte_area > 0:
                    area_fraction = dendrite_area / electrolyte_area * 100.0
                else:
                    area_fraction = np.nan
                    
                results.append([
                    file_index,
                    os.path.basename(current_file),
                    time,
                    dendrite_area,
                    area_fraction
                ])
            except Exception as e:
                st.warning(f"⚠️ Error processing {os.path.basename(current_file)} at time {time}: {e}")
        
        progress_bar.progress((file_index + 1) / total_files)
        
    status_text.text("✅ Finished!")
    
    # Display Results
    if results:
        df = pd.DataFrame(
            results,
            columns=[
                "File_Index",
                "File",
                "Time",
                "Dendrite_Area",
                "Area_Fraction_percent"
            ]
        )
        
        st.subheader("📊 Results Data")
        st.dataframe(df)
        
        # Excel Download Button
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
        excel_data = excel_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Excel File",
            data=excel_data,
            file_name="dendrite_area_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Matplotlib Plot
        st.subheader("📈 Dendrite Area Fraction vs Time")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            df["Time"],
            df["Area_Fraction_percent"],
            marker="o",
            markersize=2,
            linewidth=1.5
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Dendrite Area Fraction (%)")
        ax.set_title("Dendrite Area Fraction vs Time")
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.warning("⚠️ No results were generated. Please check your files and parameters.")
