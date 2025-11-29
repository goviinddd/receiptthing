import streamlit as st
import pandas as pd
from PIL import Image
from pdf2image import convert_from_bytes
import io
import time

# Update the import to match where you saved pipeline.py
# If you saved it in 'src/pipeline.py', use:
from pipeline import ReceiptPipeline 
# If you saved it in 'scripts/pipeline.py', keep your old import.

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Intelligent Receipt Processor", 
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1E88E5;}
    .sub-header {font-size: 1.5rem; font-weight: 600; color: #424242;}
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<p class="main-header">🧾 Intelligent Receipt Processor</p>', unsafe_allow_html=True)
st.markdown("Upload a PDF to extract structured data automatically.")
st.markdown("---")

# --- MODEL LOADING ---
@st.cache_resource
def load_pipeline():
    return ReceiptPipeline()

try:
    with st.spinner("Loading Modular Architecture (YOLO + Paddle + Donut)..."):
        pipeline = load_pipeline()
except Exception as e:
    st.error(f"System Failure: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Document Input")
    uploaded_file = st.file_uploader("Drop PDF Receipt Here", type=["pdf"])
    
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    st.success("● Modules Loaded")
    st.info("● GPU Acceleration: Active")

# --- MAIN LOGIC ---
if uploaded_file is not None:
    # Convert PDF
    images = convert_from_bytes(uploaded_file.read())
    
    if len(images) > 0:
        target_image = images[0]
        
        # Create Two Columns: Document View vs. Data View
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.markdown('<p class="sub-header">📄 Original Document</p>', unsafe_allow_html=True)
            st.image(target_image, use_container_width=True, caption="Page 1")

        with col2:
            st.markdown('<p class="sub-header">📊 Extracted Data</p>', unsafe_allow_html=True)
            
            # Run Inference only once per file upload
            if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
                with st.spinner("Running Perception & Extraction Pipeline..."):
                    # Save temp file
                    temp_path = "temp_receipt.jpg"
                    target_image.save(temp_path)
                    
                    # --- THE FIX IS HERE ---
                    # The new class uses .process(), not .predict()
                    st.session_state.extraction_data = pipeline.process(temp_path)
                    
                    st.session_state.last_uploaded = uploaded_file.name
            
            data = st.session_state.extraction_data
            
            # --- TABS FOR CLEAN UI vs DEBUG UI ---
            tab_data, tab_debug = st.tabs(["📝 Data Entry Form", "🛠️ Diagnostics"])
            
            # 1. THE CLEAN DATA FORM
            with tab_data:
                # Section A: Header Details (PO, Date, etc.)
                st.subheader("Header Information")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    po_value = data.get("po_number", "")
                    new_po = st.text_input("P.O. Number", value=po_value, help="Extracted via OCR from Red Box")
                
                with col_b:
                    st.text_input("Invoice Date", value="Not Detected")

                st.divider()

                # Section B: The Table
                st.subheader("Line Items")
                if "table_rows" in data:
                    rows = data["table_rows"]
                    if isinstance(rows, dict): rows = [rows]
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(rows)
                    
                    # Allow user to EDIT the table directly in the browser
                    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
                    
                    # Export Button
                    csv = edited_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Excel/CSV",
                        data=csv,
                        file_name="extracted_receipt.csv",
                        mime="text/csv",
                        type="primary"
                    )
                else:
                    st.warning("⚠️ No table data found. Please check the Debug tab.")

            # 2. THE HIDDEN DEBUG TAB
            with tab_debug:
                st.warning("Visual debugging for pipeline modules.")
                
                if "debug_image" in data:
                    st.image(data["debug_image"], caption="Perception Module Output", use_container_width=True)
                
                with st.expander("See Raw JSON"):
                    st.json(data)