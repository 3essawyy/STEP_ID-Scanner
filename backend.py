import cv2
import numpy as np
import os
import easyocr
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from commonfunctions import * # --- CONFIGURATION ---
TARGET_IMG_SIZE = (32, 32)

# --- NOISE & CONTRAST ---
def is_impulsive_noise(img, threshold=0.1, black_range=(0, 9), white_range=(246, 255)):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total_pixels = img.size
    is_pepper = (img >= black_range[0]) & (img <= black_range[1])
    is_salt = (img >= white_range[0]) & (img <= white_range[1])
    noise_mask = is_pepper | is_salt
    if (np.sum(noise_mask) / total_pixels) < threshold:
        return img, False 
    k = int(3 + (np.sum(noise_mask) / total_pixels) * 10)
    if k % 2 == 0: k += 1
    median = cv2.medianBlur(img, min(max(k, 3), 9))
    treated = img.copy()
    treated[noise_mask] = median[noise_mask]
    return treated, True

def is_random_noise(img):
    if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21), True

def enhance_contrast_clahe(img):
    if len(img.shape) == 3:
        l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)

# --- ALIGNMENT & EXTRACTION ---
def align_images_sift(img_to_align, reference_path):
    img2 = cv2.imread(reference_path)
    if img2 is None: return img_to_align
    gray1 = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY) if len(img_to_align.shape)==3 else img_to_align
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape)==3 else img2
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    if len(good) > 10:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return cv2.warpPerspective(img_to_align, M, (img2.shape[1], img2.shape[0]))
    return img_to_align

def extract_name_and_digits(aligned_image):
    name_roi = aligned_image[205:355, 100:1300]
    code_roi = aligned_image[404:514, 640:975]
    daf3_roi = aligned_image[500:610, 350:970]
    
    def get_digits(roi, limit):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape)==3 else roi
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if h > 15 and w > 5:
                if w > 0.8 * h:
                    candidates.extend([(x, y, w//2, h, (w//2)*h), (x+w//2, y, w//2, h, (w//2)*h)])
                else:
                    candidates.append((x, y, w, h, w*h))
                    
        top_c = sorted(candidates, key=lambda x: x[4], reverse=True)[:limit]
        final_c = sorted(top_c, key=lambda x: x[0])
        return [roi[y:y+h, x:x+w] for x, y, w, h, _ in final_c]

    return name_roi, get_digits(code_roi, 7), get_digits(daf3_roi, 14), daf3_roi, code_roi

# --- MODEL TRAINING ---
def extract_hog_features(img):
    if len(img.shape)==3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1], TARGET_IMG_SIZE)
    return cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9).compute(img).flatten()

def load_and_train_model(train_path):
    if not os.path.exists(train_path): return None
    feats, lbls = [], []
    map_lbl = {chr(97+i): str(i) for i in range(10)} 
    
    for f in os.listdir(train_path):
        if f.endswith(('.jpg','.png')) and f[0].lower() in map_lbl:
            feats.append(extract_hog_features(cv2.imread(os.path.join(train_path, f))))
            lbls.append(map_lbl[f[0].lower()])
            
    clf = Pipeline([('s', StandardScaler()), ('m', LinearSVC(random_state=42, dual=False))])
    clf.fit(feats, lbls)
    return clf

# --- ACCURACY CALCULATION ---
def calculate_pipeline_accuracy(true_file_path, extracted_file_path):
    try:
        df_true = pd.read_excel(true_file_path)
        df_extracted = pd.read_excel(extracted_file_path)
    except Exception as e:
        return f"Error loading files: {e}"

    df_true.columns = df_true.columns.str.strip()
    df_extracted.columns = df_extracted.columns.str.strip()

    min_len = min(len(df_true), len(df_extracted))
    df_true = df_true.iloc[:min_len].reset_index(drop=True)
    df_extracted = df_extracted.iloc[:min_len].reset_index(drop=True)

    report = [f"--- Accuracy Report ({min_len} rows) ---"]
    
    # Flags to track which row was correct/incorrect
    row_flags = {
        'Code': [False] * min_len,
        'Daf3': [False] * min_len,
        'Name': [False] * min_len
    }
    scores = {}

    for col in ['Code', 'Daf3', 'Name']:
        if col not in df_true.columns or col not in df_extracted.columns:
            scores[col] = 0.0
            report.append(f"Missing Column: {col}")
            continue

        true_vals = df_true[col].astype(str).fillna('').str.strip().str.replace(r'\.0$', '', regex=True)
        ext_vals = df_extracted[col].astype(str).fillna('').str.strip().str.replace(r'\.0$', '', regex=True)

        if col == 'Name':
            col_scores = []
            for i, (t, e) in enumerate(zip(true_vals, ext_vals)):
                t_no, e_no = t.replace(" ", ""), e.replace(" ", "")
                is_correct = False
                
                # Loose Match Logic
                if t == e or (t_no == e_no and abs(len(t)-len(e)) <= 1):
                    col_scores.append(1.0)
                    is_correct = True
                else:
                    t_set, e_set = set(t.split()), set(e.split())
                    if not t_set: val = 1.0 if not e_set else 0.0
                    else: val = len(t_set & e_set) / len(t_set)
                    col_scores.append(val)
                    if val >= 1.0: is_correct = True # Consider 100% word match as correct

                row_flags['Name'][i] = is_correct
            
            accuracy = np.mean(col_scores) * 100
        
        else:
            # Exact Match Logic
            matches = (true_vals == ext_vals)
            accuracy = matches.mean() * 100
            for i, m in enumerate(matches):
                row_flags[col][i] = m

        scores[col] = accuracy
        report.append(f"{col} Accuracy: {accuracy:.2f}%")

    # --- CALCULATE ROW ACCURACY & LIST FAILURES ---
    perfect_rows = 0
    failed_ids = []

    # Try to find an ID column for reporting
    id_col = 'Student ID' if 'Student ID' in df_extracted.columns else df_extracted.columns[0]
    
    for i in range(min_len):
        c_ok = row_flags['Code'][i]
        d_ok = row_flags['Daf3'][i]
        n_ok = row_flags['Name'][i]
        
        if c_ok and d_ok and n_ok:
            perfect_rows += 1
        else:
            # Identify what failed
            reasons = []
            if not c_ok: reasons.append("Code")
            if not d_ok: reasons.append("Daf3")
            if not n_ok: reasons.append("Name")
            
            # Get the ID of the student
            student_id = df_extracted.iloc[i][id_col]
            failed_ids.append(f"ID: {student_id} | Failed: {', '.join(reasons)}")

    row_acc = (perfect_rows / min_len) * 100 if min_len > 0 else 0
    
    report.append("-" * 20)
    report.append(f"ROW ACCURACY (Perfect Matches): {row_acc:.2f}%")
    report.append(f"AVERAGE COLUMN ACCURACY: {(sum(scores.values())/len(scores) if scores else 0):.2f}%")
    report.append("-" * 20)
    
    # Append the list of failed IDs
    if failed_ids:
        report.append("\n--- PROBLEMATIC IDS ---")
        report.append("\n".join(failed_ids))
    else:
        report.append("\nNo Failures! All rows match perfectly.")

    return "\n".join(report)
# --- SINGLE PROCESSOR ---
def process_single_id(img, svm, reader, ref_path):
    img, _ = is_impulsive_noise(img)
    img = align_images_sift(img, ref_path)
    img, _ = is_random_noise(img)
    img = enhance_contrast_clahe(img)
    
    name_img, code_digits, daf3_digits, daf3_full, code_full = extract_name_and_digits(img)
    
    # --- RESTORED: Your Original Loop Logic ---
    digit_preds = []
    for digit_img in code_digits:
        feat = extract_hog_features(digit_img)
        pred = svm.predict([feat])[0]
        digit_preds.append(str(pred))
    code_txt = ''.join(digit_preds)

    daf3_preds = []
    for d_img in daf3_digits:
        feat = extract_hog_features(d_img)
        pred = svm.predict([feat])[0]
        daf3_preds.append(str(pred))
    daf3_txt = ''.join(daf3_preds)
    
    name_txt = " ".join(reader.readtext(name_img, detail=0, paragraph=True)).strip()

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