import cv2
import numpy as np
import os
import easyocr
import pandas as pd
import csv
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- 1. ROBUST PATH CONFIGURATION ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_TO_TRAIN_DATASET = os.path.join(BACKEND_DIR, "train_digits")
PATH_TO_RAW_IDS = os.path.join(BACKEND_DIR, "Raw_IDs")
REF_IMG_PATH = os.path.join(BACKEND_DIR, "Raw_IDs", "ID14.jpg")
NAME_LEXICON_CSV = os.path.join(BACKEND_DIR, "name_labels.csv") 

# --- CONFIGURATION ---
TARGET_IMG_SIZE = (32, 32)
random_seed = 42 

# --- EASYOCR CONFIGURATION ---
EASYOCR_READTEXT_KWARGS = dict(
    detail=1,
    paragraph=False,
    decoder="beamsearch",
    beamWidth=5,
    batch_size=1,
    text_threshold=0.55,
    low_text=0.30,
    link_threshold=0.35,
    contrast_ths=0.08,
    adjust_contrast=0.7,
    mag_ratio=2.0,
)

# --- LEXICON HELPERS ---
_NAME_LEXICON = None

def _load_name_lexicon():
    global _NAME_LEXICON
    if _NAME_LEXICON is not None:
        return _NAME_LEXICON
    lex = []
    if os.path.exists(NAME_LEXICON_CSV):
        try:
            with open(NAME_LEXICON_CSV, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    v = (row.get("transcription") or row.get("Name") or "").strip()
                    if v:
                        lex.append(v)
            print(f"[BACKEND] Loaded lexicon with {len(lex)} entries.")
        except Exception as e:
            print(f"[BACKEND] Error loading lexicon: {e}")
            lex = []
    
    _NAME_LEXICON = lex
    return _NAME_LEXICON

def _norm_dl(s: str) -> str:
    return (s or "").replace("د", "X").replace("ل", "X")

def _levenshtein(a: str, b: str) -> int:
    a, b = a or "", b or ""
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        ca = a[i - 1]
        for j in range(1, m + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]

def _correct_with_lexicon_dl(pred: str) -> str:
    lex = _load_name_lexicon()
    if not pred or not lex:
        return pred
    p = _norm_dl(pred)
    best_name, best_dist = pred, 10**9
    for cand in lex:
        d = _levenshtein(p, _norm_dl(cand))
        if d < best_dist:
            best_dist, best_name = d, cand
    
    tol = max(2, int(0.18 * max(len(best_name), 1)))
    return best_name if best_dist <= tol else pred

# --- NOISE & CONTRAST ---
def is_impulsive_noise(img, threshold=0.1, black_range=(0, 9), white_range=(246, 255)):
    if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total_pixels = img.size
    is_pepper = (img >= black_range[0]) & (img <= black_range[1])
    is_salt = (img >= white_range[0]) & (img <= white_range[1])
    noise_mask = is_pepper | is_salt
    if (np.sum(noise_mask) / total_pixels) < threshold: return img, False 
    k = int(3 + (np.sum(noise_mask) / total_pixels) * 10)
    if k % 2 == 0: k += 1
    median = cv2.medianBlur(img, min(max(k, 3), 9))
    treated = img.copy()
    treated[noise_mask] = median[noise_mask]
    return treated, True

def is_random_noise(img, threshold=0.1):
    if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21), True

def enhance_contrast_clahe(img, clip_limit=2.0, tile_size=(8, 8)):
    if len(img.shape) == 3:
        l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
        l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size).apply(img)

# --- ALIGNMENT ---
def align_images_sift(img_to_align, reference_path):
    if not os.path.exists(reference_path) and os.path.exists(REF_IMG_PATH): reference_path = REF_IMG_PATH
    img1 = img_to_align
    img2 = cv2.imread(reference_path)
    if img2 is None: return img_to_align
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
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
    return img1

# --- EXTRACTION ---
def extract_name_and_digits(aligned_image):
    name_coords = (100, 205, 1200, 150)
    code_coords = (640, 404, 335, 110)
    daf3_coords = (350, 500, 620, 110)
    
    nx, ny, nw, nh = name_coords
    cx, cy, cw, ch = code_coords
    dx, dy, dw, dh = daf3_coords
    
    # Extract ROIs
    name_img = aligned_image[ny:ny+nh, nx:nx+nw]
    code_roi = aligned_image[cy:cy+ch, cx:cx+cw]
    daf3_roi = aligned_image[dy:dy+dh, dx:dx+dw]
    daf3_full = aligned_image[dy:dy+dh, dx:dx+dw]  # Keep full ROI for saving
    code_full = aligned_image[cy:cy+ch, cx:cx+cw]  # Keep full ROI for saving
    
    # --- Helper Function to Process Any ROI ---
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
    daf3_digits = process_roi_digits(daf3_roi, digit_limit=14)

    return name_img, code_digits, daf3_digits, daf3_full, code_full

# --- OCR HELPER (MERGED LOGIC) ---
def extractname(image_input, reader):
    """
    Reads text using EasyOCR with Beamsearch and Lexicon Correction.
    """
    try:
        image_to_read = None
        # 1. PREPARE IMAGE (JPG SIMULATION)
        if isinstance(image_input, str):
            image_to_read = image_input
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3:
                image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            success, encoded_img = cv2.imencode('.jpg', image_input)
            if success:
                image_to_read = encoded_img.tobytes()
            else:
                image_to_read = image_input
        else:
            return ""

        # 2. READ TEXT
        results = reader.readtext(image_to_read, **EASYOCR_READTEXT_KWARGS)

        # 3. PARSE & SORT
        items = []
        for r in results:
            try:
                bbox, text, conf = r
                if not isinstance(text, str): continue
                text = text.strip()
                if len(text) < 2: continue
                cx = float(np.mean([p[0] for p in bbox]))
                items.append((cx, text))
            except: continue

        items.sort(key=lambda t: t[0], reverse=True)
        joined = " ".join([t for _, t in items]).strip()

        # 4. LEXICON CORRECTION
        out = _correct_with_lexicon_dl(joined)
        return out.strip()
        
    except Exception as e:
        print(f"EasyOCR Error: {e}")
        return ""

# --- SAVING & TRAINING ---
def save_student_name(student_id, name_img, output_folder="extracted_names"):
    full_out_path = os.path.join(BACKEND_DIR, output_folder)
    if not os.path.exists(full_out_path): os.makedirs(full_out_path)
    cv2.imwrite(os.path.join(full_out_path, f"{student_id}_name.jpg"), name_img)

def save_split_digits(student_id, digit_imgs, output_folder="extracted_digits"):
    full_out_path = os.path.join(BACKEND_DIR, output_folder, student_id)
    if not os.path.exists(full_out_path): os.makedirs(full_out_path)
    for index, digit_img in enumerate(digit_imgs):
        cv2.imwrite(os.path.join(full_out_path, f"digit_{index}.jpg"), digit_img)

def extract_hog_features(img):
    if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = cv2.resize(img, (32, 32)) 
    return cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9).compute(img).flatten()

def train_SVM_robust():
    if not os.path.exists(PATH_TO_TRAIN_DATASET): return None
    
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

    if not features: return None
    
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
def _norm_arabic_variants(s: str) -> str:
    # Treat: ي == ى, and ه == ة (normalize to ي and ه)
    return (s or "").replace("ى", "ي").replace("ة", "ه")

def calculate_pipeline_accuracy(true_file_path, extracted_file_path):
    try:
        df_true = pd.read_excel(true_file_path)
        df_extracted = pd.read_excel(extracted_file_path)
    except Exception as e:
        print(f"Error loading files: {e}")
        return 0.0

    df_true.columns = df_true.columns.str.strip()
    df_extracted.columns = df_extracted.columns.str.strip()

    min_len = min(len(df_true), len(df_extracted))
    df_true = df_true.iloc[:min_len].reset_index(drop=True)
    df_extracted = df_extracted.iloc[:min_len].reset_index(drop=True)

    columns_to_check = ['Code', 'Daf3', 'Name']
    scores = {}
    
    # Track correctness for each row
    row_flags = {
        'Code': [False] * min_len,
        'Daf3': [False] * min_len,
        'Name': [False] * min_len
    }

    print(f"--- Accuracy Report (Checking {min_len} rows) ---\n")

    for col in columns_to_check:
        if col not in df_true.columns or col not in df_extracted.columns:
            print(f"Error: Column '{col}' missing.")
            print(f"   Available in True File: {df_true.columns.tolist()}")
            print(f"   Available in Extracted: {df_extracted.columns.tolist()}")
            scores[col] = 0.0
            continue

        true_series = df_true[col].astype(str).fillna('')
        extracted_series = df_extracted[col].astype(str).fillna('')

        true_clean = true_series.str.strip().str.replace(r'\.0$', '', regex=True)
        extracted_clean = extracted_series.str.strip().str.replace(r'\.0$', '', regex=True)

        if col == 'Name':
            row_scores = []

            for i, (t_val, e_val) in enumerate(zip(true_clean, extracted_clean)):
                t_val = _norm_arabic_variants(t_val)
                e_val = _norm_arabic_variants(e_val)

                t_nospace = t_val.replace(" ", "")
                e_nospace = e_val.replace(" ", "")
                
                is_correct = False

                if t_val == e_val:
                    row_scores.append(1.0)
                    is_correct = True
                elif (t_nospace == e_nospace) and (abs(len(t_val) - len(e_val)) <= 1):
                    row_scores.append(1.0)
                    is_correct = True
                else:
                    t_words = set(t_val.split())
                    e_words = set(e_val.split())
                    if len(t_words) == 0:
                        val = 1.0 if len(e_words) == 0 else 0.0
                        row_scores.append(val)
                        if val == 1.0: is_correct = True
                    else:
                        common = t_words.intersection(e_words)
                        score = len(common) / len(t_words)
                        row_scores.append(score)
                        if score >= 1.0: is_correct = True
                
                row_flags['Name'][i] = is_correct

            accuracy = np.mean(row_scores) * 100
        else:
            matches = (true_clean == extracted_clean)
            accuracy = (matches.sum() / len(matches)) * 100
            
            # Track correctness for Code and Daf3
            for i, m in enumerate(matches):
                row_flags[col][i] = m

        scores[col] = accuracy
        print(f"{col} Accuracy: {accuracy:.2f}%")

    average_accuracy = (sum(scores.values()) / len(scores)) if scores else 0.0

    # --- CALCULATE ROW ACCURACY ---
    perfect_rows = 0
    failed_ids = []
    
    # Try to find ID column for reporting
    id_col = 'Student ID' if 'Student ID' in df_extracted.columns else df_extracted.columns[0]

    for i in range(min_len):
        c_ok = row_flags['Code'][i]
        d_ok = row_flags['Daf3'][i]
        n_ok = row_flags['Name'][i]
        
        if c_ok and d_ok and n_ok:
            perfect_rows += 1
        else:
            # Generate failure report for this row
            reasons = []
            if not c_ok: reasons.append("Code")
            if not d_ok: reasons.append("Daf3")
            if not n_ok: reasons.append("Name")
            sid = df_extracted.iloc[i][id_col]
            failed_ids.append(f"ID: {sid} | Failed: {', '.join(reasons)}")

    row_accuracy = (perfect_rows / min_len) * 100 if min_len > 0 else 0.0

    # Build report string
    report_lines = []
    report_lines.append("\n--------------------------------")
    report_lines.append(f"AVERAGE ACCURACY: {average_accuracy:.2f}%")
    report_lines.append(f"ROW ACCURACY: {row_accuracy:.2f}%")
    report_lines.append("--------------------------------")
    
    if failed_ids:
        report_lines.append("\n--- PROBLEMATIC IDS ---")
        report_lines.extend(failed_ids)
    else:
        report_lines.append("\nNo Failures! All rows match perfectly.")

    report_str = "\n".join(report_lines)
    print(report_str)
    
    return report_str

# --- PIPELINES ---
def process_single_id(img, SVMclassifier, reader, ref_path):
    if not os.path.exists(ref_path): ref_path = REF_IMG_PATH
    img, _ = is_impulsive_noise(img)
    img = align_images_sift(img, ref_path)
    img, _ = is_random_noise(img)
    img = enhance_contrast_clahe(img)
    name_img, code_digits, daf3_digits, daf3_full, code_full = extract_name_and_digits(img)
    
    code_txt = ''.join([str(SVMclassifier.predict([extract_hog_features(d)])[0]) for d in code_digits])
    daf3_txt = ''.join([str(SVMclassifier.predict([extract_hog_features(d)])[0]) for d in daf3_digits])
    name_txt = extractname(name_img, reader)

    return {
        "aligned_image": img, "name_image": name_img, "daf3_image": daf3_full, "code_image": code_full,
        "digit_imgs": code_digits, "daf3_digit_imgs": daf3_digits,
        "name_text": name_txt, "code_text": code_txt, "daf3_text": daf3_txt
    }

def main_pipeline():
    print("Initializing EasyOCR (GPU=True)...")
    reader = easyocr.Reader(['ar', 'en'], gpu=True)
    svm = train_SVM_robust()
    if not svm or not os.path.exists(PATH_TO_RAW_IDS): return
    
    data = []
    for fn in os.listdir(PATH_TO_RAW_IDS):
        path = os.path.join(PATH_TO_RAW_IDS, fn)
        if not os.path.isfile(path): continue
        if not fn.lower().endswith(('.jpg', '.png', '.jpeg')): continue
        
        # Load Raw
        raw = cv2.imread(path)
        res = process_single_id(raw, svm, reader, REF_IMG_PATH)
        
        # Save Crops (Needed for main_pipeline behavior)
        sid = os.path.splitext(fn)[0]
        save_student_name(sid, res['name_image'])
        save_split_digits(sid, res['digit_imgs'])
        save_split_digits(f"{sid}_daf3", res['daf3_digit_imgs'], "extracted_daf3_digits")
        
        data.append({"Student ID": sid, "Name": res['name_text'], "Code": res['code_text'], "Daf3": res['daf3_text']})
        print(f"Processed {sid}")
        
    df = pd.DataFrame(data)
    out_path = os.path.join(BACKEND_DIR, "Extracted_Results.xlsx")
    df.to_excel(out_path, index=False)
    print(f"Batch processing done. Saved to {out_path}")
    
    # Run Accuracy
    true_path = os.path.join(BACKEND_DIR, "True_Results.xlsx")
    if os.path.exists(true_path):
        print(calculate_pipeline_accuracy(true_path, out_path))

if __name__ == "__main__":
    main_pipeline()