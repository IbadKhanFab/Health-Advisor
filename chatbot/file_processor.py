import os
import io
import base64
import tempfile
import PyPDF2
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict
from .config import Config
import re

logger = logging.getLogger("HealthChatbot")

class FileProcessor:
    @staticmethod
    def get_file_type(file_path: str) -> str:
        """Determine file type based on extension"""
        if file_path.lower().endswith('.pdf'):
            return "pdf"
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return "image"
        elif file_path.lower().endswith('.txt'):
            return "text"
        raise ValueError("Unsupported file format. Please upload PDF, PNG, JPG, or TXT.")

    @staticmethod
    def should_use_multimodal(file_path: str, file_content: str) -> bool:
        """Determine if multimodal processing should be used"""
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return True
        if file_path.lower().endswith('.pdf'):
            if not file_content or len(file_content.strip()) < 200:
                return True
            if not any(keyword in file_content.lower() for keyword in Config.MEDICAL_KEYWORDS):
                return True
        return False

    @staticmethod
    def extract_text_from_file(file_path: str) -> Tuple[str, str]:
        """Extract text from file with enhanced methods, returns (text, method_used)"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            if file_path.lower().endswith('.pdf'):
                return FileProcessor._extract_text_from_pdf(file_path)
            elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                return FileProcessor._extract_text_from_image(file_path)
            elif file_path.lower().endswith('.txt'):
                return FileProcessor._extract_text_from_txt(file_path)
            else:
                raise ValueError("Unsupported file format")
                
        except Exception as e:
            logger.error(f"Error in _extract_text_from_file: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error processing file: {str(e)}")

    @staticmethod
    def _extract_text_from_pdf(file_path: str) -> Tuple[str, str]:
        """Enhanced PDF text extraction with multiple methods and fallbacks"""
        extracted_text = []
        methods_used = []
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                pytext = "\n".join([page.extract_text() or "" for page in reader.pages])
                if pytext.strip():
                    extracted_text.append(pytext)
                    methods_used.append("PyPDF2")
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {str(e)}")
        
        try:
            doc = fitz.open(file_path)
            mutext = ""
            for page in doc:
                mutext += page.get_text()
            if mutext.strip():
                extracted_text.append(mutext)
                methods_used.append("PyMuPDF")
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {str(e)}")
        
        if not extracted_text or len(" ".join(extracted_text).strip()) < 100:
            try:
                doc = fitz.open(file_path)
                ocr_text = []
                
                for page_num in range(min(len(doc), 3)):  
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img)
                    if not text.strip():
                        img_np = np.array(img)
                        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                    cv2.THRESH_BINARY, 11, 2)
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                            temp_filename = temp.name
                            cv2.imwrite(temp_filename, binary)
                            text = pytesseract.image_to_string(Image.open(temp_filename))
                            os.unlink(temp_filename)
                    
                    ocr_text.append(text)
                
                ocr_result = "\n\n".join(ocr_text)
                if ocr_result.strip():
                    extracted_text.append(ocr_result)
                    methods_used.append("OCR")
            except Exception as e:
                logger.warning(f"OCR for PDF failed: {str(e)}")

        if extracted_text:
            best_text = max(extracted_text, key=len)
            method = "+".join(methods_used)
            return best_text, method
        return "", "failed"

    @staticmethod
    def _extract_text_from_image(file_path: str) -> Tuple[str, str]:
        """Enhanced image text extraction using preprocessing for better OCR results"""
        try:
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("Unable to read image file")
    
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            methods = []
            results = []
            
            standard_text = pytesseract.image_to_string(Image.open(file_path))
            if standard_text.strip():
                results.append(standard_text)
                methods.append("standard")
            _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                temp_filename = temp.name
                cv2.imwrite(temp_filename, binary_otsu)
                otsu_text = pytesseract.image_to_string(Image.open(temp_filename))
                os.unlink(temp_filename)
                if otsu_text.strip():
                    results.append(otsu_text)
                    methods.append("otsu")
            adaptive_binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                temp_filename = temp.name
                cv2.imwrite(temp_filename, adaptive_binary)
                adaptive_text = pytesseract.image_to_string(Image.open(temp_filename))
                os.unlink(temp_filename)
                if adaptive_text.strip():
                    results.append(adaptive_text)
                    methods.append("adaptive")
                    
            return results[0], methods[0] if results else ("", "failed")
            
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Image processing error: {str(e)}")

    @staticmethod
    def _extract_text_from_txt(file_path: str) -> Tuple[str, str]:
        """Extract text from txt file with encoding fallbacks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read(), "utf-8"
        except UnicodeDecodeError:
            encodings = ['latin-1', 'iso-8859-1', 'windows-1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read(), encoding
                except UnicodeDecodeError:
                    continue
            raise RuntimeError("Unable to decode text file with common encodings")
        except Exception as e:
            raise RuntimeError(f"Text file error: {str(e)}")

    @staticmethod
    def prepare_file_for_multimodal(file_path: str) -> List[Dict]:
        """Prepare file for multimodal processing by converting to images"""
        images = []
        
        try:
            if file_path.lower().endswith('.pdf'):
                doc = fitz.open(file_path)
                for page_num in range(min(len(doc), 5)): 
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
                    img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img_byte_arr = io.BytesIO()
                    img_data.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    images.append({
                        "inlineData": {
                            "data": base64.b64encode(img_byte_arr).decode('utf-8'),
                            "mimeType": "image/png"
                        }
                    })
            
            elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                with open(file_path, "rb") as img_file:
                    img_bytes = img_file.read()
                    mime_type = f"image/{os.path.splitext(file_path)[1][1:]}"
                    if mime_type == "image/jpg":
                        mime_type = "image/jpeg"
                    
                    images.append({
                        "inlineData": {
                            "data": base64.b64encode(img_bytes).decode('utf-8'),
                            "mimeType": mime_type
                        }
                    })
            
            return images
            
        except Exception as e:
            logger.error(f"Error preparing file for multimodal: {str(e)}", exc_info=True)
            raise Exception(f"Error preparing document for visual analysis: {str(e)}")