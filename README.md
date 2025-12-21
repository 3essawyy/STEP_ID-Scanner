# STEP ID Scanner 📄🔍

An end-to-end **Python OCR project** for extracting student information from **college ID cards**, with a focus on **Arabic text recognition**. The project combines image preprocessing, OCR engines, and basic machine learning to read, clean, and organize data from scanned ID images.

---

## 🚀 Project Overview

This project was developed to automatically read student data (**Name, Student code, and Payment ID**) from ID card images. It is designed to handle **Arabic text**, Digits, common OCR challenges (noise, low contrast), and real-world scanned images.

Key goals:

- Improve OCR accuracy on Arabic ID cards
- Apply image preprocessing to enhance text clarity
- Extract and organize results into structured formats (e.g., tables)

---

## 🧠 Features

- ✅ Arabic text recognition
- 🖼️ Image preprocessing (grayscale, denoising, sharpening, CLAHE)
- 🔤 Arabic OCR using **EasyOCR**
- 📊 Data handling with **Pandas**
- 🤖 ML pipeline (SVM) for classification/validation for digits
- 🧪 Interactive result inspection (row-by-row image + extracted data)

---
## 🖥️ User Interface (Streamlit Dashboard)

The project includes a **Streamlit-based interactive web UI** that allows running the OCR pipeline without using the command line.

### 🔹 Single ID Processing
- Upload a single ID image
- Displays:
  - Original image
  - Aligned image
  - Cropped regions (Name, Code, Payment ID)
  - Individual digit segments
- Shows extracted:
  - Arabic name
  - Student code
  - Payment ID

### 🔹 Batch Folder Processing
- Processes all images inside the `Raw_IDs/` directory
- Live processing feed with:
  - Original vs aligned images
  - Extracted fields per ID
- Automatically exports results to **Excel**
- Optional accuracy report using a provided `True Results.xlsx`
   
---

## 🛠️ Tech Stack

- **Python 3.9+**
- **OpenCV** – image processing
- **EasyOCR** – primary OCR engine (Arabic support)
- **NumPy** – numerical operations
- **Pandas** – data storage and analysis
- **Scikit-learn** – SVM & preprocessing
- **Streamlit** (optional) – simple UI

---

## 📂 Project Structure

```
STEP_ID-Scanner/
│
├── backend.py          # Core OCR & processing logic
├── app.py              # Streamlit app (optional UI)
├── STEP_Scanner.ipynb  # Notebook for experimentation & testing
├── Raw_IDs/            # ID card images
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/STEP_ID-Scanner.git
cd STEP_ID-Scanner
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Install Tesseract OCR**

- Windows: Install from the official Tesseract installer and add it to PATH
- Linux:

```bash
sudo apt install tesseract-ocr tesseract-ocr-ara
```

---

## ▶️ Usage

### Run backend processing

```bash
python backend.py
```

### Run Streamlit app (optional)

```bash
streamlit run app.py
```

### Notebook

Open `STEP_Scanner.ipynb` to:

- Test preprocessing steps
- Compare OCR outputs
- Debug Arabic text extraction

---

## 🧪 OCR Strategy (Arabic)

- Preprocessing improves contrast and suppresses noise
- CLAHE enhances local details without over-amplifying noise
- Arabic text normalization (Alif normalization, trimming spaces)
- EasyOCR used as the primary engine due to stronger Arabic performance

---

## 📊 Output

- Extracted student data displayed in console / UI
- Results can be exported to **CSV / Excel** for further analysis



---

## 🚧 Limitations & Future Work

- Improve OCR accuracy on low-resolution images
- Add deep-learning-based text detection
- Support more ID layouts
- Automate dataset labeling
- Add confidence scoring per field

##
