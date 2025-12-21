import cv2
import numpy as np
import os
import easyocr
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- 1. ROBUST PATH CONFIGURATION ---
# This gets the exact folder where backend.py is located
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# Define absolute paths based on the backend file location
PATH_TO_TRAIN_DATASET = os.path.join(BACKEND_DIR, "train_digits")
PATH_TO_RAW_IDS = os.path.join(BACKEND_DIR, "Raw_IDs")
REF_IMG_PATH = os.path.join(BACKEND_DIR, "Raw_IDs", "ID14.jpg")

# Debugging: Print where we are strictly looking
print(f"[BACKEND] Backend File is at: {BACKEND_DIR}")
print(f"[BACKEND] Looking for 'train_digits' at: {PATH_TO_TRAIN_DATASET}")
print(f"[BACKEND] Looking for 'Raw_IDs' at: {PATH_TO_RAW_IDS}")

# --- CONFIGURATION ---
TARGET_IMG_SIZE = (32, 32)
random_seed = 42 

# --- NOISE & CONTRAST (EXACT ORIGINALS) ---

def is_impulsive_noise(img, threshold=0.1, black_range=(0, 9), white_range=(246, 255)):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    total_pixels = img.size
    is_pepper = (img >= black_range[0]) & (img <= black_range[1])
    is_salt = (img >= white_range[0]) & (img <= white_range[1])
    
    noise_mask = is_pepper | is_salt
    num_noise_pixels = np.sum(noise_mask)
    prop = num_noise_pixels / total_pixels

    if prop < threshold:
        return img, False 

    k = int(3 + prop * 10)
    if k % 2 == 0: k += 1
    k = min(max(k, 3), 9)

    median_filtered = cv2.medianBlur(img, k)
    treated_img = img.copy()
    treated_img[noise_mask] = median_filtered[noise_mask]

    return treated_img, True

def is_random_noise(img, threshold=0.1):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    treated_img = cv2.fastNlMeansDenoising(
        img, 
        None, 
        h=10, 
        templateWindowSize=7, 
        searchWindowSize=21
    )
    
    return treated_img, True

def enhance_contrast_clahe(img, clip_limit=2.0, tile_size=(8, 8)):
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(img)

# --- ALIGNMENT (EXACT ORIGINAL) ---

def align_images_sift(img_to_align, reference_path):
    # Ensure we use the robust path if reference_path is just a filename
    if not os.path.exists(reference_path) and os.path.exists(REF_IMG_PATH):
        reference_path = REF_IMG_PATH

    img1 = img_to_align
    img2 = cv2.imread(reference_path) # Train Image
    
    if img2 is None: 
        return img_to_align

    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1 

    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2

    sift = cv2.SIFT_create() 
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = img2.shape[:2]
        aligned_img = cv2.warpPerspective(img1, M, (w, h))
        return aligned_img
    else:
        print(f"Not enough matches found: {len(good_matches)}/10")
        return img1

# --- EXTRACTION ---

def extract_name_and_digits(aligned_image):
    name_coords = (100, 205, 1200, 150)
    code_coords = (640, 404, 335, 110)
    daf3_coords = (350, 500, 620, 110)
    
    nx, ny, nw, nh = name_coords
    cx, cy, cw, ch = code_coords
    dx, dy, dw, dh = daf3_coords
    
    name_img = aligned_image[ny:ny+nh, nx:nx+nw]
    code_roi = aligned_image[cy:cy+ch, cx:cx+cw]
    daf3_img = aligned_image[dy:dy+dh, dx:dx+dw]
    
    def process_roi_digits(roi_img, digit_limit):
        if len(roi_img.shape) == 3:
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_img
        
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if h > 15 and w > 5:
                if w > 0.8 * h: 
                    half_w = w // 2
                    candidates.append((x, y, half_w, h, half_w * h))
                    candidates.append((x + half_w, y, half_w, h, half_w * h))
                else:
                    candidates.append((x, y, w, h, area))
        
        candidates = sorted(candidates, key=lambda c: c[4], reverse=True)[:digit_limit]
        final_candidates = sorted(candidates, key=lambda c: c[0])
        
        cropped_digits = []
        for (x, y, w, h, area) in final_candidates:
            digit_crop = roi_img[y:y+h, x:x+w]
            cropped_digits.append(digit_crop)
            
        return cropped_digits

    code_digits = process_roi_digits(code_roi, digit_limit=7)
    daf3_digits = process_roi_digits(daf3_img, digit_limit=14)

    return name_img, code_digits, daf3_digits, daf3_img, code_roi
# --- OCR HELPER (Updated for Exact Notebook Match) ---
def extractname(image_input, reader):
    """
    Reads text using EasyOCR.
    1. If input is a path, read directly (Batch Mode).
    2. If input is an array (Single Mode), compress to JPG in memory first
       to simulate the 'save to disk' step from the notebook.
    """
    try:
        # CASE A: It's a file path (Batch Mode) -> Pass path string directly
        if isinstance(image_input, str):
            results = reader.readtext(image_input, detail=0, paragraph=True)
        
        # CASE B: It's an image array (Single Mode)
        elif isinstance(image_input, np.ndarray):
            # 1. Convert BGR to RGB (Just in case)
            if len(image_input.shape) == 3:
                image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            
            # 2. CRITICAL: Encode to JPG in memory to simulate file saving
            # This adds the exact same compression artifacts as cv2.imwrite
            success, encoded_img = cv2.imencode('.jpg', image_input)
            
            if success:
                # Pass the compressed bytes to EasyOCR
                results = reader.readtext(encoded_img.tobytes(), detail=0, paragraph=True)
            else:
                # Fallback to raw array if encoding fails
                results = reader.readtext(image_input, detail=0, paragraph=True)
            
        else:
            return ""

        # Join results exactly like the notebook
        full_name = " ".join(results)
        return full_name.strip()
        
    except Exception as e:
        print(f"EasyOCR Error: {e}")
        return ""
# --- SAVING HELPERS ---
def save_student_name(student_id, name_img, output_folder="extracted_names"):
    # Ensure output folder is relative to backend to avoid path errors
    full_out_path = os.path.join(BACKEND_DIR, output_folder)
    if not os.path.exists(full_out_path): os.makedirs(full_out_path)
    cv2.imwrite(os.path.join(full_out_path, f"{student_id}_name.jpg"), name_img)

def save_split_digits(prefix, digit_imgs, output_folder="extracted_digits"):
    full_out_path = os.path.join(BACKEND_DIR, output_folder)
    if not os.path.exists(full_out_path): os.makedirs(full_out_path)
    for idx, d_img in enumerate(digit_imgs):
        cv2.imwrite(os.path.join(full_out_path, f"{prefix}_digit_{idx}.jpg"), d_img)

# --- TRAINING (UPDATED TO USE ROBUST PATHS) ---

def extract_hog_features(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = cv2.resize(img, (32, 32)) 
    
    hog = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)
    h = hog.compute(img)
    return h.flatten()

def train_SVM_robust():
    # USES THE GLOBALLY DEFINED ROBUST PATH
    if not os.path.exists(PATH_TO_TRAIN_DATASET):
        print(f"[ERROR] Could not find dataset at: {PATH_TO_TRAIN_DATASET}")
        return None

    label_map = {
        'a': '0', 'b': '1', 'c': '2', 'd': '3', 'e': '4', 
        'f': '5', 'g': '6', 'h': '7', 'i': '8', 'j': '9'
    }
    
    features = []
    labels = []
    
    img_filenames = os.listdir(PATH_TO_TRAIN_DATASET)
    print(f"Loading {len(img_filenames)} training images...")

    for fn in img_filenames:
        if not fn.lower().endswith(('.jpg', '.png')):
            continue

        prefix = fn[0].lower()
        if prefix in label_map:
            labels.append(label_map[prefix])
            path = os.path.join(PATH_TO_TRAIN_DATASET, fn)
            img = cv2.imread(path)
            features.append(extract_hog_features(img))
    
    if len(features) == 0:
        print("[ERROR] No images found! Check your folder content.")
        return None

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', LinearSVC(random_state=42, max_iter=5000, dual=False))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=random_seed
    )
    
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Training Complete. Validation Accuracy: {accuracy*100:.2f}%")
    
    return clf

# --- ACCURACY CALCULATION ---
def calculate_pipeline_accuracy(true_file_path, extracted_file_path):
    try:
        df_true = pd.read_excel(true_file_path)
        df_extracted = pd.read_excel(extracted_file_path)
    except Exception as e: return f"Error: {e}"

    df_true.columns = df_true.columns.str.strip()
    df_extracted.columns = df_extracted.columns.str.strip()
    min_len = min(len(df_true), len(df_extracted))
    df_true, df_extracted = df_true.iloc[:min_len], df_extracted.iloc[:min_len]

    report = [f"--- Accuracy Report ({min_len} rows) ---"]
    row_flags = {'Code': [False]*min_len, 'Daf3': [False]*min_len, 'Name': [False]*min_len}
    scores = {}

    for col in ['Code', 'Daf3', 'Name']:
        if col not in df_true.columns or col not in df_extracted.columns: continue
        t_vals = df_true[col].astype(str).fillna('').str.strip().str.replace(r'\.0$', '', regex=True)
        e_vals = df_extracted[col].astype(str).fillna('').str.strip().str.replace(r'\.0$', '', regex=True)

        if col == 'Name':
            c_scores = []
            for i, (t, e) in enumerate(zip(t_vals, e_vals)):
                t_n, e_n = t.replace(" ", ""), e.replace(" ", "")
                if t == e or (t_n == e_n and abs(len(t)-len(e)) <= 1): val = 1.0
                else:
                    ts, es = set(t.split()), set(e.split())
                    val = (len(ts & es) / len(ts)) if ts else (1.0 if not es else 0.0)
                c_scores.append(val)
                row_flags['Name'][i] = (val >= 1.0)
            acc = np.mean(c_scores) * 100
        else:
            matches = (t_vals == e_vals)
            acc = matches.mean() * 100
            for i, m in enumerate(matches): row_flags[col][i] = m
        scores[col] = acc
        report.append(f"{col} Accuracy: {acc:.2f}%")

    perfect = 0
    failed = []
    id_col = 'Student ID' if 'Student ID' in df_extracted.columns else df_extracted.columns[0]
    for i in range(min_len):
        if row_flags['Code'][i] and row_flags['Daf3'][i] and row_flags['Name'][i]: perfect += 1
        else:
            reasons = [k for k in ['Code','Daf3','Name'] if not row_flags[k][i]]
            failed.append(f"ID: {df_extracted.iloc[i][id_col]} | Failed: {', '.join(reasons)}")

    report.append("-" * 20)
    report.append(f"ROW ACCURACY: {(perfect/min_len)*100:.2f}%")
    if failed: report.append("\n--- PROBLEMATIC IDS ---\n" + "\n".join(failed))
    else: report.append("\nNo Failures!")
    return "\n".join(report)

# --- SINGLE PROCESSOR (For Dashboard) ---

def process_single_id(img, SVMclassifier, reader, ref_path):
    # Use global REF path if local one is bad
    if not os.path.exists(ref_path): ref_path = REF_IMG_PATH

    img, _ = is_impulsive_noise(img)
    img = align_images_sift(img, ref_path)
    img, _ = is_random_noise(img)
    img = enhance_contrast_clahe(img)
    
    # Unpack 5 values
    name_img, code_digits, daf3_digits, daf3_full, code_full = extract_name_and_digits(img)
    
    digit_preds = []
    for digit_img in code_digits:
        feat = extract_hog_features(digit_img)
        pred = SVMclassifier.predict([feat])[0]
        digit_preds.append(str(pred))
    code_txt = ''.join(digit_preds)

    daf3_preds = []
    for d_img in daf3_digits:
        feat = extract_hog_features(d_img)
        pred = SVMclassifier.predict([feat])[0]
        daf3_preds.append(str(pred))
    daf3_txt = ''.join(daf3_preds)
    
    name_txt = extractname(name_img, reader)

    return {
        "aligned_image": img,
        "name_image": name_img,
        "daf3_image": daf3_full,
        "code_image": code_full,
        "digit_imgs": code_digits,
        "daf3_digit_imgs": daf3_digits,
        "name_text": name_txt,
        "code_text": code_txt,
        "daf3_text": daf3_txt
    }

# --- MAIN PIPELINE (UPDATED FOR BATCH) ---

def main_pipeline():
    # Use GLOBAL ROBUST PATHS
    path_to_dataset = PATH_TO_RAW_IDS
    refrence_image_path = REF_IMG_PATH

    # Initialize Reader once here (Using GPU=True as per your snippet)
    print("Initializing EasyOCR (GPU=True)...")
    reader = easyocr.Reader(['ar', 'en'], gpu=True)

    # Ensure the classifier is trained
    SVMclassifier = train_SVM_robust()
    if SVMclassifier is None: 
        print("Classifier could not be trained. Aborting.")
        return

    data_for_excel = []

    if not os.path.exists(path_to_dataset):
        print(f"Directory not found: {path_to_dataset}")
        return

    print("Starting Batch Processing...")
    
    for i in os.listdir(path_to_dataset):
        if not i.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        
        img_path = os.path.join(path_to_dataset, i)
        raw_img = cv2.imread(img_path)
        clean_img, is_impulsive = is_impulsive_noise(raw_img)
        aligned_img = align_images_sift(clean_img, refrence_image_path)
        clean_img2, is_random = is_random_noise(aligned_img)
        enhanced_img = enhance_contrast_clahe(clean_img2)
        
        # NOTE: Unpack 5 values (Backend compatible), use _ to ignore extra dashboard images
        name_img, digit_imgs, daf3_digits, _, _ = extract_name_and_digits(enhanced_img)
        
        student_id = os.path.splitext(i)[0]

        # Use updated save functions (saves relative to backend.py)
        save_student_name(student_id, name_img)
        save_split_digits(student_id, digit_imgs)
        save_split_digits(f"{student_id}_daf3", daf3_digits, output_folder="extracted_daf3_digits")

        digit_preds = []
        for digit_img in digit_imgs:
            feat = extract_hog_features(digit_img)
            pred = SVMclassifier.predict([feat])[0]
            digit_preds.append(str(pred))
        code_str = ''.join(digit_preds)

        daf3_preds = []
        for d_img in daf3_digits:
            feat = extract_hog_features(d_img)
            pred = SVMclassifier.predict([feat])[0]
            daf3_preds.append(str(pred))
        daf3_str = ''.join(daf3_preds)

        # Extract Name - Pointing to the file we just saved
        saved_name_path = os.path.join(BACKEND_DIR, 'extracted_names', f'{student_id}_name.jpg')
        name_text = extractname(saved_name_path, reader)

        data_for_excel.append({
            "Student ID": student_id,   
            "Name": name_text,
            "Code": code_str,
            "Daf3": daf3_str,
        })
        print(f"Processed {student_id}")

    if data_for_excel:
        df = pd.DataFrame(data_for_excel)
        print("Processing Complete. Results:")
        print(df[['Student ID', 'Name', 'Code', 'Daf3']])

        output_file = os.path.join(BACKEND_DIR, "Extracted_Results.xlsx")
        df.to_excel(output_file, index=False)
        print(f"Excel file saved to: {output_file}")
        
        true_results_path = os.path.join(BACKEND_DIR, 'True_Results.xlsx')
        if os.path.exists(true_results_path):
            acc_report = calculate_pipeline_accuracy(true_results_path, output_file)
            print(acc_report)
        else:
            print(f"Skipping accuracy check: '{true_results_path}' not found.")

if __name__ == "__main__":
    main_pipeline()