# import os
# import io
# import base64
# import tempfile
# from dotenv import load_dotenv
# from pymongo import MongoClient
# from datetime import datetime
# from typing import Dict, Any, List, Tuple, Optional
# import google.generativeai as genai
# import certifi
# import PyPDF2
# from PIL import Image
# import pytesseract
# import fitz  # PyMuPDF
# import cv2
# import numpy as np
# import logging
# import json
# from bson import ObjectId


# logging.basicConfig(level=logging.INFO, 
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     handlers=[logging.FileHandler("health_chatbot.log"), 
#                               logging.StreamHandler()])
# logger = logging.getLogger("HealthChatbot")

# load_dotenv()

# class HealthChatbot:
#     def __init__(self, test_mode: bool = False):
#         self.test_mode = test_mode
#         if not test_mode:
#             self.gemini_api_key = os.getenv("GEMINI_API_KEY")
#             self.mongo_uri = os.getenv("MONGO_URI")
#             self.db_name = os.getenv("MONGO_DB_NAME", "health_chatbot")

#             if not self.gemini_api_key:
#                 raise ValueError("GEMINI_API_KEY not found in environment variables")
#             if not self.mongo_uri:
#                 raise ValueError("MONGO_URI not found in environment variables")

#             genai.configure(api_key=self.gemini_api_key)
#             self.text_model = genai.GenerativeModel("gemini-1.5-flash-latest")
#             self.multimodal_model = genai.GenerativeModel("gemini-1.5-pro-latest")

#             self.client = MongoClient(self.mongo_uri, tlsCAFile=certifi.where())
#             self.db = self.client[self.db_name]
#             self.conversations = self.db["conversations"]
#             self.chats = self.db["chats"]  # New collection for chat sessions
            
#             logger.info("HealthChatbot initialized successfully")

#     def start_new_chat(self, person_id: str) -> str:
#         """Start a new chat session for a person"""
#         try:
#             chat_data = {
#                 "person_id": person_id,
#                 "created_at": datetime.utcnow(),
#                 "updated_at": datetime.utcnow(),
#                 "status": "active",
#                 "conversation_ids": []
#             }
#             result = self.chats.insert_one(chat_data)
#             return str(result.inserted_id)
#         except Exception as e:
#             logger.error(f"Error starting new chat: {str(e)}", exc_info=True)
#             raise RuntimeError("Could not start new chat session")

#     def get_chat_history(self, chat_id: str, limit: int = 10) -> List[Dict]:
#         """Get conversation history for a chat session"""
#         try:
#             chat = self.chats.find_one({"_id": ObjectId(chat_id)})
#             if not chat:
#                 return []
            
#             conversation_ids = chat.get("conversation_ids", [])
#             conversations = list(self.conversations.find(
#                 {"_id": {"$in": conversation_ids}},
#                 {"user_input": 1, "ai_response": 1, "timestamp": 1, "_id": 0}
#             ).sort("timestamp", -1).limit(limit))
            
#             return conversations[::-1]  # Return in chronological order
#         except Exception as e:
#             logger.error(f"Error getting chat history: {str(e)}", exc_info=True)
#             return []

#     def analyze_health_input(self, person_id: str, chat_id: str, user_input: str, file_path: str = None) -> Dict[str, Any]:
#         """
#         Analyze health input with support for text and file uploads.
#         Uses multimodal processing for PDFs and images when needed.
#         """
#         try:
#             if self._is_emergency(user_input):
#                 logger.warning(f"Emergency keywords detected: {user_input}")
#                 return {
#                     "status": "emergency",
#                     "response": self._get_emergency_response(),
#                     "timestamp": datetime.utcnow().isoformat()
#                 }
            
#             # Get chat history for context
#             chat_history = self.get_chat_history(chat_id)
#             context = "\n".join(
#                 [f"User: {conv['user_input']}\nAI: {conv['ai_response']}" 
#                  for conv in chat_history]
#             )
            
#             file_content = None
#             image_data = None
#             response = None
            
#             if file_path:
#                 logger.info(f"Processing file: {file_path}")
#                 file_type = self._get_file_type(file_path)
#                 file_content, extraction_method = self._extract_text_from_file(file_path)
#                 logger.info(f"Text extraction method: {extraction_method}")
#                 logger.info(f"Extracted text length: {len(file_content) if file_content else 0} chars")
#                 if self._should_use_multimodal(file_path, file_content):
#                     logger.info("Using multimodal approach")
#                     image_data = self._prepare_file_for_multimodal(file_path)
#                     response = self._process_multimodal(context, user_input, file_content, image_data)
#                 else:
#                     logger.info("Using text-only approach")
#                     prompt = self._build_prompt(context, user_input, file_content)
#                     response = self._call_gemini_api(prompt)
#             else:
#                 prompt = self._build_prompt(context, user_input)
#                 response = self._call_gemini_api(prompt)
            
#             # Store conversation and update chat
#             conversation_id = self._store_conversation(
#                 person_id=person_id,
#                 chat_id=chat_id,
#                 user_input=user_input,
#                 file_path=file_path,
#                 file_content=file_content,
#                 ai_response=response
#             )
            
#             # Update chat with new conversation
#             self.chats.update_one(
#                 {"_id": ObjectId(chat_id)},
#                 {
#                     "$push": {"conversation_ids": conversation_id},
#                     "$set": {"updated_at": datetime.utcnow()}
#                 }
#             )
            
#             logger.info(f"Analysis complete. Conversation ID: {conversation_id}")
            
#             return {
#                 "status": "success",
#                 "response": response,
#                 "conversation_id": str(conversation_id),
#                 "chat_id": chat_id,
#                 "timestamp": datetime.utcnow().isoformat()
#             }

#         except Exception as e:
#             logger.error(f"Error in analyze_health_input: {str(e)}", exc_info=True)
            
#             if self._is_emergency(user_input):
#                 return {
#                     "status": "emergency",
#                     "response": self._get_emergency_response(),
#                     "timestamp": datetime.utcnow().isoformat()
#                 }

#             return {
#                 "status": "error",
#                 "response": f"System temporarily unavailable. Please seek medical attention if this is urgent.\nError: {str(e)}",
#                 "timestamp": datetime.utcnow().isoformat()
#             }

    
#     def _is_emergency(self, text: str) -> bool:
#         """
#         Check if the input contains emergency keywords
#         Uses word boundaries to avoid false positives like 'diesease' triggering on 'die'
#         """
#         emergency_phrases = [
#             r"\bdie\b", r"\bdying\b", r"\bheart attack\b", r"\bstroke\b", 
#             r"\bcan't breathe\b", r"\bcannot breathe\b", r"\bsuicide\b", 
#             r"\bkill myself\b", r"\bemergency\b", r"\b911\b", r"\bambulance\b"
#         ]
        
#         # Use word boundaries for emergency keywords to avoid false positives
#         import re
#         for phrase in emergency_phrases:
#             if re.search(phrase, text.lower()):
#                 logger.warning(f"Emergency keyword detected: {phrase}")
#                 return True
        
#         return False
        
#     def _get_file_type(self, file_path: str) -> str:
#         """Determine file type based on extension"""
#         if file_path.lower().endswith('.pdf'):
#             return "pdf"
#         elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#             return "image"
#         elif file_path.lower().endswith('.txt'):
#             return "text"
#         else:
#             raise ValueError("Unsupported file format. Please upload PDF, PNG, JPG, or TXT.")
    
#     def _should_use_multimodal(self, file_path: str, file_content: str) -> bool:
#         """Determine if multimodal processing should be used"""
#         if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#             return True
#         if file_path.lower().endswith('.pdf'):
#             if not file_content or len(file_content.strip()) < 200:
#                 return True
#             medical_keywords = ["blood", "test", "result", "count", "level", "mg", "dl", "mmol"]
#             if not any(keyword in file_content.lower() for keyword in medical_keywords):
#                 return True
        
#         return False

#     def _build_prompt(self, context: str, symptoms: str, file_content: str = None) -> str:
#         """Build prompt with context from previous conversations"""
#         prompt = ""
        
#         if context:
#             prompt += f"Previous conversation context:\n{context}\n\n"
        
#         prompt += f"""Current patient reports: "{symptoms}"\n\n"""
        
#         if file_content:
#             prompt += f"""Test results provided:\n{file_content}\n\n"""
        
#         prompt += """As a medical AI assistant, please provide a comprehensive health analysis including:

#     1. Possible conditions (ranked by likelihood based on the information provided)
#     2. Medication suggestions (both over-the-counter and prescription options that might be appropriate)
#     3. Exercise routines tailored to the patient's condition
#     4. Meditation and stress-management techniques relevant to their symptoms
#     5. Nutritional guidance with specific food recommendations and restrictions
#     6. Immediate recommended actions 
#     7. When to seek emergency care
#     8. Follow-up recommendations
#     9. Standard medical disclaimer

#     Note: Be precise and thorough in your analysis of the provided test results, mentioning specific values that are out of range and their potential implications. For medication suggestions, exercise routines, meditation techniques, and nutritional guidance, provide specific, actionable recommendations that are safe and appropriate given the patient's reported symptoms and test results."""
        
#         return prompt
#     def _store_conversation(self, person_id: str, chat_id: str, user_input: str, 
#                 ai_response: str, file_path: str = None, 
#                 file_content: str = None) -> ObjectId:
#         """Store conversation in MongoDB"""
#         try:
#             conversation = {
#                 "person_id": person_id,
#                 "chat_id": ObjectId(chat_id),
#                 "user_input": user_input,
#                 "file_path": os.path.basename(file_path) if file_path else None,
#                 "file_content_sample": file_content[:500] if file_content else None,
#                 "file_content_length": len(file_content) if file_content else 0,
#                 "ai_response": ai_response,
#                 "timestamp": datetime.utcnow()
#             }
#             result = self.conversations.insert_one(conversation)
#             return result.inserted_id
#         except Exception as e:
#             logger.error(f"Error storing conversation: {str(e)}", exc_info=True)
#             raise RuntimeError("Could not store conversation")

#     def _call_gemini_api(self, prompt: str) -> str:
#         """Call Gemini text API"""
#         try:
#             response = self.text_model.generate_content(prompt)
#             return response.text.strip() if response and response.text else "No response from Gemini"
#         except Exception as e:
#             logger.error(f"Gemini API error: {str(e)}", exc_info=True)
#             raise Exception(f"Gemini API error: {str(e)}")

#     def _process_multimodal(self, symptoms: str, file_content: str, image_data: List[Dict]) -> str:
#         """Process input using multimodal capabilities with comprehensive health guidance"""
#         try:
#             # Create multimodal prompt
#             prompt_parts = []
            
#             # Add text instructions
#             text_prompt = f"""Patient reports: "{symptoms}"\n\n"""
#             if file_content and len(file_content.strip()) > 0:
#                 text_prompt += f"""Some text was extracted from the document:\n{file_content}\n\n"""
            
#             text_prompt += """Please analyze the provided medical report/image and deliver a comprehensive health plan including:

#     1. Identifying the type of medical document and extracting all relevant medical values
#     2. Identifying abnormal values and their potential clinical significance
#     3. Possible conditions (ranked by likelihood)
#     4. Medication suggestions (include specific over-the-counter options and types of prescriptions that might be beneficial)
#     5. Exercise routines (provide specific exercises, duration, frequency, and intensity appropriate for the condition)
#     6. Meditation and stress-management techniques (include specific practices, duration, and frequency)
#     7. Nutritional guidance (recommend specific foods to consume and avoid, meal timing, and portion guidance)
#     8. Immediate recommended actions
#     9. When to seek emergency care
#     10. Follow-up recommendations
#     11. Standard medical disclaimer

#     Be thorough in your analysis of any visible test results, mentioning specific values that are out of range and their potential implications. For all recommendations (medications, exercises, meditation, nutrition), provide detailed, actionable guidance that considers the patient's specific condition."""
            
#             prompt_parts.append(text_prompt)
            
#             # Add images
#             for img_data in image_data:
#                 prompt_parts.append(img_data)
            
#             # Generate response
#             response = self.multimodal_model.generate_content(prompt_parts)
#             return response.text.strip() if response and response.text else "No response from Gemini"
            
#         except Exception as e:
#             logger.error(f"Multimodal processing error: {str(e)}", exc_info=True)
#             raise Exception(f"Error processing document visually: {str(e)}")
#     def _prepare_file_for_multimodal(self, file_path: str) -> List[Dict]:
#         """Prepare file for multimodal processing by converting to images"""
#         images = []
        
#         try:
#             if file_path.lower().endswith('.pdf'):
#                 doc = fitz.open(file_path)
#                 for page_num in range(min(len(doc), 5)): 
#                     page = doc.load_page(page_num)
#                     pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
#                     img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#                     img_byte_arr = io.BytesIO()
#                     img_data.save(img_byte_arr, format='PNG')
#                     img_byte_arr = img_byte_arr.getvalue()
                    
#                     images.append({
#                         "inlineData": {
#                             "data": base64.b64encode(img_byte_arr).decode('utf-8'),
#                             "mimeType": "image/png"
#                         }
#                     })
            
#             elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 # Load image directly
#                 with open(file_path, "rb") as img_file:
#                     img_bytes = img_file.read()
#                     mime_type = f"image/{os.path.splitext(file_path)[1][1:]}"
#                     if mime_type == "image/jpg":
#                         mime_type = "image/jpeg"
                    
#                     images.append({
#                         "inlineData": {
#                             "data": base64.b64encode(img_bytes).decode('utf-8'),
#                             "mimeType": mime_type
#                         }
#                     })
            
#             return images
            
#         except Exception as e:
#             logger.error(f"Error preparing file for multimodal: {str(e)}", exc_info=True)
#             raise Exception(f"Error preparing document for visual analysis: {str(e)}")

#     def _get_emergency_response(self) -> str:
#         """Standard emergency response"""
#         return """🚨 EMERGENCY MEDICAL ALERT 🚨

# Your symptoms require immediate attention:
# 1. CALL EMERGENCY SERVICES NOW (911/112)
# 2. Do not ignore these symptoms
# 3. If alone, contact someone who can help
# 4. Stay on the line with emergency operator

# Common emergency symptoms:
# - Feeling like you might die
# - Chest pain
# - Severe difficulty breathing
# - Sudden severe pain
# - Loss of consciousness

# This is not an AI response - this is standard emergency protocol."""



#     def _extract_text_from_file(self, file_path: str) -> Tuple[str, str]:
#         """Extract text from file with enhanced methods, returns (text, method_used)"""
#         try:
#             if not os.path.exists(file_path):
#                 raise FileNotFoundError(f"File not found: {file_path}")
                
#             if file_path.lower().endswith('.pdf'):
#                 return self._extract_text_from_pdf(file_path)
#             elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 return self._extract_text_from_image(file_path)
#             elif file_path.lower().endswith('.txt'):
#                 return self._extract_text_from_txt(file_path)
#             else:
#                 raise ValueError("Unsupported file format. Please upload PDF, PNG, JPG, or TXT.")
                
#         except Exception as e:
#             logger.error(f"Error in _extract_text_from_file: {str(e)}", exc_info=True)
#             raise RuntimeError(f"Error processing file: {str(e)}")

#     def _extract_text_from_pdf(self, file_path: str) -> Tuple[str, str]:
#         """Enhanced PDF text extraction with multiple methods and fallbacks"""
#         extracted_text = []
#         methods_used = []
        
#         try:
#             with open(file_path, 'rb') as file:
#                 reader = PyPDF2.PdfReader(file)
#                 pytext = "\n".join([page.extract_text() or "" for page in reader.pages])
#                 if pytext.strip():
#                     extracted_text.append(pytext)
#                     methods_used.append("PyPDF2")
#         except Exception as e:
#             logger.warning(f"PyPDF2 extraction failed: {str(e)}")
#         try:
#             doc = fitz.open(file_path)
#             mutext = ""
#             for page in doc:
#                 mutext += page.get_text()
#             if mutext.strip():
#                 extracted_text.append(mutext)
#                 methods_used.append("PyMuPDF")
#         except Exception as e:
#             logger.warning(f"PyMuPDF extraction failed: {str(e)}")
#         if not extracted_text or len(" ".join(extracted_text).strip()) < 100:
#             try:
#                 doc = fitz.open(file_path)
#                 ocr_text = []
                
#                 for page_num in range(min(len(doc), 3)):  
#                     page = doc.load_page(page_num)
#                     pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
#                     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#                     text = pytesseract.image_to_string(img)
#                     if not text.strip():
#                         img_np = np.array(img)
#                         gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
#                         binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                                       cv2.THRESH_BINARY, 11, 2)
#                         with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
#                             temp_filename = temp.name
#                             cv2.imwrite(temp_filename, binary)
#                             text = pytesseract.image_to_string(Image.open(temp_filename))
#                             os.unlink(temp_filename)
                    
#                     ocr_text.append(text)
                
#                 ocr_result = "\n\n".join(ocr_text)
#                 if ocr_result.strip():
#                     extracted_text.append(ocr_result)
#                     methods_used.append("OCR")
#             except Exception as e:
#                 logger.warning(f"OCR for PDF failed: {str(e)}")

#         if extracted_text:
#             best_text = max(extracted_text, key=len)
#             method = "+".join(methods_used)
#             return best_text, method
#         else:
#             return "", "failed"

#     def _extract_text_from_image(self, file_path: str) -> Tuple[str, str]:
#         """Enhanced image text extraction using preprocessing for better OCR results"""
#         try:
        
#             img = cv2.imread(file_path)
#             if img is None:
#                 raise ValueError("Unable to read image file")
    
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
#             methods = []
#             results = []
#             standard_text = pytesseract.image_to_string(Image.open(file_path))
#             if standard_text.strip():
#                 results.append(standard_text)
#                 methods.append("standard")
           
#             _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#             with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
#                 temp_filename = temp.name
#                 cv2.imwrite(temp_filename, binary_otsu)
#                 otsu_text = pytesseract.image_to_string(Image.open(temp_filename))
#                 os.unlink(temp_filename)
#                 if otsu_text.strip():
#                     results.append(otsu_text)
#                     methods.append("otsu")
            
#             # Method 3: Adaptive thresholding
#             adaptive_binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                                    cv2.THRESH_BINARY, 11, 2)
#             with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
#                 temp_filename = temp.name
#                 cv2.imwrite(temp_filename, adaptive_binary)
#                 adaptive_text = pytesseract.image_to_string(Image.open(temp_filename))
#                 os.unlink(temp_filename)
#                 if adaptive_text.strip():
#                     results.append(adaptive_text)
#                     methods.append("adaptive")
                    
#             # Return the best result (longest text)
#             if results:
#                 best_idx = results.index(max(results, key=len))
#                 return results[best_idx], methods[best_idx]
            
#             return "", "failed"
            
#         except Exception as e:
#             logger.error(f"Image processing error: {str(e)}", exc_info=True)
#             raise RuntimeError(f"Image processing error: {str(e)}")

#     def _extract_text_from_txt(self, file_path: str) -> Tuple[str, str]:
#         """Extract text from txt file with encoding fallbacks"""
#         try:
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 return file.read(), "utf-8"
#         except UnicodeDecodeError:
#             # Try different encodings if UTF-8 fails
#             encodings = ['latin-1', 'iso-8859-1', 'windows-1252']
#             for encoding in encodings:
#                 try:
#                     with open(file_path, 'r', encoding=encoding) as file:
#                         return file.read(), encoding
#                 except UnicodeDecodeError:
#                     continue
#             raise RuntimeError("Unable to decode text file with common encodings")
#         except Exception as e:
#             raise RuntimeError(f"Text file error: {str(e)}")

# def main():
#     print("Health Chatbot (type 'quit' to exit)")
#     try:
#         chatbot = HealthChatbot(test_mode=False)
        
#         # Get or create person ID (in a real app, this would come from authentication)
#         person_id = input("Enter your person ID (or leave blank to create new): ").strip()
#         if not person_id:
#             person_id = str(ObjectId())  # Generate a new ID
#             print(f"Your new person ID is: {person_id}")
        
#         # Start a new chat session
#         chat_id = chatbot.start_new_chat(person_id)
#         print(f"New chat session started. Chat ID: {chat_id}")
        
#         while True:
#             try:
#                 print("\nChoose input type:")
#                 print("[1] Text description")
#                 print("[2] File upload (PDF/PNG/JPG/TXT)")
#                 print("[h] View chat history")
#                 print("[n] Start new chat")
#                 print("[q] Quit")
#                 choice = input("> ").strip().lower()
                
#                 if choice in ['q', 'quit', 'exit']:
#                     break
#                 elif choice == 'h':
#                     history = chatbot.get_chat_history(chat_id)
#                     print("\nChat History:")
#                     for i, conv in enumerate(history, 1):
#                         print(f"\n{i}. {conv['timestamp']}")
#                         print(f"User: {conv['user_input']}")
#                         print(f"AI: {conv['ai_response']}")
#                     continue
#                 elif choice == 'n':
#                     chat_id = chatbot.start_new_chat(person_id)
#                     print(f"\nNew chat session started. Chat ID: {chat_id}")
#                     continue

#                 if choice == '1':
#                     symptoms = input("\nDescribe your symptoms or health concern: ").strip()
#                     file_path = None
#                 elif choice == '2':
#                     file_path = input("\nEnter path to your file: ").strip()
                    
#                     if not os.path.exists(file_path):
#                         print(f"\n Error: File '{file_path}' does not exist")
#                         continue
                        
#                     # Try to extract some text to preview
#                     try:
#                         file_content, extraction_method = chatbot._extract_text_from_file(file_path)
#                         print(f"\n File processed using {extraction_method}")
                        
#                         if file_content and len(file_content.strip()) > 0:
#                             preview = file_content[:300] + "..." if len(file_content) > 300 else file_content
#                             print("\nExtracted text preview:")
#                             print("-" * 40)
#                             print(preview)
#                             print("-" * 40)
                            
#                             if len(file_content.strip()) < 100:
#                                 print("\n Limited text extracted. Will use visual analysis as well.")
#                         else:
#                             print("\n No text could be extracted. Will use visual analysis.")
                            
#                         symptoms = input("\nDescribe your symptoms or concerns related to this report: ").strip()
#                     except Exception as e:
#                         print(f"\n❌ Error processing file: {str(e)}")
#                         continue
#                 else:
#                     print("Invalid choice. Please enter 1, 2, h, n, or q")
#                     continue

#                 print("\nAnalyzing your input...")
#                 result = chatbot.analyze_health_input(person_id, chat_id, symptoms, file_path)

#                 print("\n" + "=" * 60)
#                 print("HEALTH RECOMMENDATION")
#                 print("=" * 60)
#                 print(result["response"])
                
#                 print(f"\nChat ID: {chat_id}")
#                 if result.get("conversation_id"):
#                     print(f"Conversation ID: {result['conversation_id']}")

#             except KeyboardInterrupt:
#                 print("\nExiting...")
#                 break
#             except Exception as e:
#                 print(f"\n Error: {str(e)}")
#                 print("Please try again or contact support.")
    
#     except Exception as e:
#         print(f"\n Critical Error: {str(e)}")
#         print("Could not initialize Health Chatbot. Please check your environment variables and dependencies.")

# if __name__ == "__main__":
#     main()























import os
import logging
from bson import ObjectId
from chatbot.health_chatbot import HealthChatbot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("health_chatbot.log"),
        logging.StreamHandler()
    ]
)

def main():
    print("Health Chatbot (type 'quit' to exit)")
    try:
        chatbot = HealthChatbot(test_mode=False)
        
        # Get or create person ID
        person_id = input("Enter your person ID (or leave blank to create new): ").strip()
        if not person_id:
            person_id = str(ObjectId())
            print(f"Your new person ID is: {person_id}")
        
        # Start a new chat session
        chat_id = chatbot.db.start_new_chat(person_id)
        print(f"New chat session started. Chat ID: {chat_id}")
        
        while True:
            try:
                print("\nChoose input type:")
                print("[1] Text description")
                print("[2] File upload (PDF/PNG/JPG/TXT)")
                print("[h] View chat history")
                print("[n] Start new chat")
                print("[q] Quit")
                choice = input("> ").strip().lower()
                
                if choice in ['q', 'quit', 'exit']:
                    break
                elif choice == 'h':
                    history = chatbot.db.get_chat_history(chat_id)
                    print("\nChat History:")
                    for i, conv in enumerate(history, 1):
                        print(f"\n{i}. {conv['timestamp']}")
                        print(f"User: {conv['user_input']}")
                        print(f"AI: {conv['ai_response']}")
                    continue
                elif choice == 'n':
                    chat_id = chatbot.db.start_new_chat(person_id)
                    print(f"\nNew chat session started. Chat ID: {chat_id}")
                    continue

                if choice == '1':
                    symptoms = input("\nDescribe your symptoms or health concern: ").strip()
                    file_path = None
                elif choice == '2':
                    file_path = input("\nEnter path to your file: ").strip()
                    
                    if not os.path.exists(file_path):
                        print(f"\nError: File '{file_path}' does not exist")
                        continue
                        
                    try:
                        file_content, method = chatbot.file_processor.extract_text_from_file(file_path)
                        print(f"\nFile processed using {method}")
                        
                        if file_content:
                            preview = file_content[:300] + ("..." if len(file_content) > 300 else "")
                            print("\nExtracted text preview:")
                            print("-" * 40)
                            print(preview)
                            print("-" * 40)
                            
                            if len(file_content.strip()) < 100:
                                print("\nLimited text extracted. Will use visual analysis.")
                        else:
                            print("\nNo text could be extracted. Will use visual analysis.")
                            
                        symptoms = input("\nDescribe your symptoms related to this report: ").strip()
                    except Exception as e:
                        print(f"\n❌ Error processing file: {str(e)}")
                        continue
                else:
                    print("Invalid choice. Please enter 1, 2, h, n, or q")
                    continue

                print("\nAnalyzing your input...")
                result = chatbot.analyze_health_input(person_id, chat_id, symptoms, file_path)
                print("HEALTH RECOMMENDATION")
                print(result["response"])
                
                print(f"\nChat ID: {chat_id}")
                if result.get("conversation_id"):
                    print(f"Conversation ID: {result['conversation_id']}")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again or contact support.")
    
    except Exception as e:
        print(f"\nCritical Error: {str(e)}")
        print("Could not initialize Health Chatbot. Please check your environment variables.")

if __name__ == "__main__":
    main()