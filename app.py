import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
import backend 
import tempfile

st.set_page_config(page_title="ID Scanner", layout="wide")

# --- USE ROBUST PATHS FROM BACKEND ---
# We use the paths we calculated in backend.py so they never break
REFERENCE_PATH = backend.REF_IMG_PATH
RAW_IDS_FOLDER = backend.PATH_TO_RAW_IDS

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        # Load Model
        svm = backend.train_SVM_robust()
        
        # --- FORCE GPU HERE ---
        # If your notebook used GPU, this MUST be True.
        print("Initializing EasyOCR with gpu=True...")
        reader = backend.easyocr.Reader(['ar', 'en'], gpu=True) 
        
        return svm, reader
    except Exception as e:
        st.error(f"Resource Error: {e}")
        return None, None

svm_model, ocr_reader = load_resources()

if not svm_model:
    st.error("Model loading failed. Ensure 'train_digits' folder is correctly placed.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("Control Panel")
mode = st.sidebar.radio("Mode", ["Single Upload", "Batch Folder Processing"])

# --- MODE 1: SINGLE ---
if mode == "Single Upload":
    st.title("Single ID Scanner")
    up_file = st.file_uploader("Upload ID", type=['jpg','png','jpeg'])
    
    if up_file:
        file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
        raw = cv2.imdecode(file_bytes, 1)
        st.image(raw, "Original", channels="BGR", width=400)
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                res = backend.process_single_id(raw, svm_model, ocr_reader, REFERENCE_PATH)
            
            aligned_img = res['aligned_image']
            img_channels = "BGR" if len(aligned_img.shape) == 3 else "RGB"

            # Result Section
            c1, c2 = st.columns(2)
            c1.image(aligned_img, "Aligned", channels=img_channels, use_container_width=True)
            
            c2.success(f"Name: {res['name_text']}")
            c2.info(f"Code: {res['code_text']}")
            c2.warning(f"Daf3: {res['daf3_text']}")
            
            st.divider()
            st.subheader("Extraction Details")
            
            d1, d2, d3 = st.columns(3)
            
            # 1. NAME
            with d1:
                st.write("**Name Crop**")
                st.image(res['name_image'], use_container_width=True)

            # 2. CODE
            with d2:
                st.write("**Code Crop**")
                if res['code_image'] is not None:
                    st.image(res['code_image'], use_container_width=True)
                
                if res['digit_imgs']:
                    st.write("Code Digits")
                    # Stack digits
                    dig_stack = np.hstack([cv2.resize(d, (32,32)) for d in res['digit_imgs']])
                    st.image(dig_stack, use_container_width=True)

            # 3. DAF3
            with d3:
                st.write("**Daf3 Crop**")
                if res['daf3_image'] is not None:
                    st.image(res['daf3_image'], use_container_width=True)

                if res['daf3_digit_imgs']:
                    st.write("Daf3 Digits")
                    # Stack digits
                    daf3_stack = np.hstack([cv2.resize(d, (32,32)) for d in res['daf3_digit_imgs']])
                    st.image(daf3_stack, use_container_width=True)

# --- MODE 2: BATCH ---
elif mode == "Batch Folder Processing":
    st.title("Batch Processing & Accuracy")
    
    # Use the robust path from backend
    folder = RAW_IDS_FOLDER
    
    if not os.path.exists(folder):
        st.error(f"Folder '{folder}' not found. Please ensure 'Raw_IDs' is next to backend.py")
        st.stop()
        
    true_file = st.file_uploader("Upload 'True Results.xlsx' for Accuracy (Optional)", type=['xlsx'])
    
    if st.button("Start Batch"):
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        if not files: st.warning("No images found."); st.stop()
        
        bar = st.progress(0)
        status = st.empty()
        results = []
        
        st.subheader("Live Processing Feed")

        for i, f in enumerate(files):
            status.text(f"Processing {i+1}/{len(files)}: {f}")
            path = os.path.join(folder, f)
            
            try:
                original_img = cv2.imread(path)
                res = backend.process_single_id(original_img, svm_model, ocr_reader, REFERENCE_PATH)
                
                aligned_img = res['aligned_image']
                res_channels = "BGR" if len(aligned_img.shape) == 3 else "RGB"
                
                with st.expander(f"✅ Processed: {f} (ID: {os.path.splitext(f)[0]})", expanded=False):
                    col_orig, col_res, col_data = st.columns([1, 1, 1])
                    
                    col_orig.image(original_img, caption="Original Input", channels="BGR", use_container_width=True)
                    col_res.image(aligned_img, caption="Aligned Output", channels=res_channels, use_container_width=True)
                    
                    col_data.markdown(f"""
                    **Name:** {res['name_text']}  
                    **Code:** `{res['code_text']}`  
                    **Daf3:** `{res['daf3_text']}`
                    """)

                row = {
                    "Student ID": os.path.splitext(f)[0],
                    "Name": res['name_text'],
                    "Code": res['code_text'],
                    "Daf3": res['daf3_text']
                }
                results.append(row)
                
            except Exception as e:
                st.error(f"Failed on {f}: {e}")
            
            bar.progress((i+1)/len(files))
            
        status.success("Batch Processing Complete!")
        
        if results:
            df = pd.DataFrame(results)
            # Safe sorting that handles IDs that might not be purely numeric
            try:
                df['SortKey'] = df['Student ID'].str.extract(r'(\d+)').astype(float)
                df = df.sort_values('SortKey').drop(columns=['SortKey'])
            except:
                pass # Skip sorting if regex fails
            
            out_path = "Extracted_Results.xlsx"
            df.to_excel(out_path, index=False)
            st.success(f"Results saved to {out_path}")
            
            st.dataframe(df)
            
            if true_file:
                st.divider()
                st.subheader("Accuracy Report")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                    tmp.write(true_file.getvalue())
                    tmp_path = tmp.name
                
                # Calculate accuracy
                report_str = backend.calculate_pipeline_accuracy(tmp_path, out_path)
                st.text(report_str)
                
                os.remove(tmp_path)