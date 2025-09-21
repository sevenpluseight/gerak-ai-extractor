from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
import pytesseract
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError
import json
import boto3
import os
from dotenv import load_dotenv
import tempfile
from concurrent.futures import ProcessPoolExecutor
import logging
from typing import List, Dict
from boto3.s3.transfer import TransferConfig
import io
from rapidfuzz import fuzz

# Load environment variables
load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", os.cpu_count()))

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Setup boto3 S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=AWS_REGION
)

# S3 multipart config for large uploads
MULTIPART_CONFIG = TransferConfig(
    multipart_threshold=50 * 1024 * 1024,  # 50MB
    max_concurrency=4,
    multipart_chunksize=25 * 1024 * 1024,  # 25MB
    use_threads=True
)

app = FastAPI(title="GerakAI PDF Extractor")
logging.basicConfig(level=logging.INFO)

# In-memory status store
file_status = {}  # key: filename, value: "processing" | "done" | "failed"

# Define the keywords to extract
KEYWORDS = [
    "Event Type Code", "Capacity", "Estimated Attendance", "Number of Gates",
    "Attendance Ratio", "Parking Capacity", "Nearby Public Transport",
    "Transport Modes Count", "Transport Max Capacity", "Transport Cancelled Count",
    "VIP Zones Flag", "Number of Restrooms", "Number of Food Courts",
    "Number of First Aid Stations", "Number of Emergency Exits", "Weather Severity",
    "Celebrity Arrival", "VIP Attending", "Road Closure Expected", "Congestion Risk"
]

def extract_keywords_from_text(text: str, keywords: List[str], threshold: int = 70) -> Dict[str, str]:
    """
    Extract lines from text that match keywords using fuzzy matching.
    """
    extracted = {}
    lines = text.split("\n")
    for line in lines:
        for keyword in keywords:
            if fuzz.partial_ratio(keyword.lower(), line.lower()) >= threshold:
                extracted[keyword] = line.strip()
    return extracted

def ocr_page(image_path: str) -> str:
    """Run OCR on a single image path"""
    from PIL import Image
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    img.close()
    return text

def process_pdf_multiprocess(pdf_path: str, filename: str):
    """Process PDF using multiprocessing, extract keywords, and upload JSON to S3"""
    file_status[filename] = "processing"
    extracted_data: List[Dict[str, str]] = []

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            images = convert_from_path(pdf_path, output_folder=tmpdir, fmt="jpeg", dpi=150)
            image_paths = [img.filename if hasattr(img, "filename") else img for img in images]

            # Run OCR using multiprocessing
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                page_texts = list(executor.map(ocr_page, image_paths))

            # Extract keywords for each page
            for text in page_texts:
                page_keywords = extract_keywords_from_text(text, KEYWORDS)
                extracted_data.append(page_keywords)

    except PDFInfoNotInstalledError:
        logging.error("pdfinfo not installed. Install poppler for PDF support.")
        file_status[filename] = "failed"
        return
    except PDFPageCountError:
        logging.error(f"Invalid PDF file: {filename}")
        file_status[filename] = "failed"
        return
    except Exception as e:
        logging.error(f"PDF processing failed for {filename}: {str(e)}")
        file_status[filename] = "failed"
        return

    # Convert to JSON
    json_data = {"filename": filename, "pages": extracted_data}

    # Remove .pdf extension for S3 key
    file_base_name = os.path.splitext(filename)[0]
    s3_key = f"{file_base_name}.json"

    # Upload JSON to S3 (multipart if large)
    try:
        json_bytes = json.dumps(json_data).encode("utf-8")
        with io.BytesIO(json_bytes) as f:
            s3_client.upload_fileobj(
                f,
                Bucket=BUCKET_NAME,
                Key=s3_key,
                Config=MULTIPART_CONFIG
            )
        logging.info(f"Uploaded {s3_key} to S3 successfully.")
        file_status[filename] = "done"
    except Exception as e:
        logging.error(f"S3 upload failed for {filename}: {str(e)}")
        file_status[filename] = "failed"
    finally:
        # Remove temp PDF
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

@app.post("/upload")
async def upload(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            content = await file.read()
            tmpfile.write(content)
            tmpfile_path = tmpfile.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save temp PDF: {str(e)}")

    # Add background task for processing
    background_tasks.add_task(process_pdf_multiprocess, tmpfile_path, file.filename)

    return {"message": "File is being processed in background", "filename": file.filename}

@app.get("/status/{filename}")
async def status(filename: str):
    status = file_status.get(filename)
    if not status:
        raise HTTPException(status_code=404, detail="File not found")
    return {"filename": filename, "status": status}
