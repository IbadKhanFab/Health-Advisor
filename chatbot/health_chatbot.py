import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from .config import Config
from .database import Database
from .file_processor import FileProcessor
from bson import ObjectId
import re

logger = logging.getLogger("HealthChatbot")

class HealthChatbot:
    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        if not test_mode:
            self._initialize_services()
            
    def _initialize_services(self):
        Config.validate()
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.text_model = genai.GenerativeModel("gemini-1.5-flash-latest")
        self.multimodal_model = genai.GenerativeModel("gemini-1.5-pro-latest")
        self.db = Database()
        self.file_processor = FileProcessor()
        logger.info("HealthChatbot initialized successfully")

    def analyze_health_input(self, person_id: str, chat_id: str, 
                           user_input: str, file_path: str = None) -> Dict[str, Any]:
        try:
            if self._is_emergency(user_input):
                logger.warning(f"Emergency keywords detected: {user_input}")
                return {
                    "status": "emergency",
                    "response": self._get_emergency_response(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Get chat history for context
            chat_history = self.db.get_chat_history(chat_id)
            context = "\n".join(
                [f"User: {conv['user_input']}\nAI: {conv['ai_response']}" 
                 for conv in chat_history]
            )
            
            file_content = None
            image_data = None
            response = None
            
            if file_path:
                logger.info(f"Processing file: {file_path}")
                file_content, extraction_method = self.file_processor.extract_text_from_file(file_path)
                logger.info(f"Text extraction method: {extraction_method}")
                logger.info(f"Extracted text length: {len(file_content) if file_content else 0} chars")
                
                if self.file_processor.should_use_multimodal(file_path, file_content):
                    logger.info("Using multimodal approach")
                    image_data = self.file_processor.prepare_file_for_multimodal(file_path)
                    response = self._process_multimodal(context, user_input, file_content, image_data)
                else:
                    logger.info("Using text-only approach")
                    prompt = self._build_prompt(context, user_input, file_content)
                    response = self._call_gemini_api(prompt)
            else:
                prompt = self._build_prompt(context, user_input)
                response = self._call_gemini_api(prompt)
            
            # Store conversation and update chat
            conversation_id = self.db.store_conversation(
                person_id=person_id,
                chat_id=chat_id,
                user_input=user_input,
                ai_response=response,
                file_path=file_path,
                file_content=file_content
            )
            
            # Update chat with new conversation
            self.db.chats.update_one(
                {"_id": ObjectId(chat_id)},
                {
                    "$push": {"conversation_ids": conversation_id},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            
            logger.info(f"Analysis complete. Conversation ID: {conversation_id}")
            
            return {
                "status": "success",
                "response": response,
                "conversation_id": str(conversation_id),
                "chat_id": chat_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in analyze_health_input: {str(e)}", exc_info=True)
            
            if self._is_emergency(user_input):
                return {
                    "status": "emergency",
                    "response": self._get_emergency_response(),
                    "timestamp": datetime.utcnow().isoformat()
                }

            return {
                "status": "error",
                "response": f"System temporarily unavailable. Please seek medical attention if this is urgent.\nError: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }

    def _is_emergency(self, text: str) -> bool:
        """Check if the input contains emergency keywords"""
        for phrase in Config.EMERGENCY_PHRASES:
            if re.search(phrase, text.lower()):
                logger.warning(f"Emergency keyword detected: {phrase}")
                return True
        return False
        
    def _build_prompt(self, context: str, symptoms: str, file_content: str = None) -> str:
        """Build prompt with interactive questioning approach"""
        prompt = ""
        
        # Handle simple greetings
        if symptoms.lower().strip() in ['hi', 'hello', 'hey', 'greetings']:
            return "Respond warmly to the greeting and ask how you can help with their health concerns today."
        
        # Handle vague symptoms
        if symptoms.lower() in ["not feeling well", "i'm sick", "i feel unwell"]:
            return """Ask 2-3 specific questions to better understand the symptoms:
            1. "Could you describe what you're feeling in more detail?"
            2. "How long have you been feeling this way?"
            3. "Are there any specific areas of your body that are bothering you?"
            Maintain a professional, caring tone."""
        
        if context:
            prompt += f"Previous conversation context:\n{context}\n\n"
        
        prompt += f"""Patient reports: "{symptoms}"\n\n"""
        
        if file_content:
            prompt += f"""Test results provided:\n{file_content}\n\n"""
        
        prompt += """Provide a structured response with:
        1. Possible conditions (ranked by likelihood)
        2. Recommended actions
        3. When to seek emergency care
        4. Follow-up recommendations
        5. Standard medical disclaimer
        
        Use clear, compassionate language suitable for a patient."""
        
        return prompt

    def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini text API"""
        try:
            response = self.text_model.generate_content(prompt)
            return response.text.strip() if response and response.text else "No response from Gemini"
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}", exc_info=True)
            raise Exception(f"Gemini API error: {str(e)}")

    def _process_multimodal(self, context: str, symptoms: str, file_content: str, image_data: List[Dict]) -> str:
        """Process input using multimodal capabilities"""
        try:
            # Handle simple greetings
            if symptoms.lower().strip() in ['hi', 'hello', 'hey', 'greetings']:
                return ("Hello! I'm your health assistant. I see you've shared a medical document. "
                      "Would you like me to review it and provide a health assessment?")

            # Handle vague symptoms
            if not symptoms.strip() or len(symptoms.strip()) < 5:
                symptoms = "The patient shared a medical document but didn't specify concerns"

            prompt_parts = []
            text_prompt = f"""Patient communication: "{symptoms}"\n\n"""
            
            if context:
                text_prompt += f"Previous context:\n{context}\n\n"
            
            if file_content:
                text_prompt += f"Extracted text from document:\n{file_content}\n\n"
            
            text_prompt += """Please analyze this medical document and provide:
            [Document Analysis]
            1. Document type and key findings
            2. Abnormal values and their significance
            
            [Recommendations]
            1. Next steps
            2. When to seek urgent care
            
            [Disclaimer]
            Include standard medical disclaimer
            
            Use clear, non-technical language."""
            
            prompt_parts.append(text_prompt)
            
            # Add images (limit to 3 pages)
            for img_data in image_data[:3]:
                prompt_parts.append(img_data)
            
            # Generate response
            response = self.multimodal_model.generate_content(
                prompt_parts,
                generation_config={"temperature": 0.3, "max_output_tokens": 2000}
            )
            
            if not response or not response.text:
                return "I couldn't generate a response for this document. Please consult a healthcare provider."
            
            cleaned_response = response.text.strip()
            if "disclaimer" not in cleaned_response.lower():
                cleaned_response += "\n\nDisclaimer: This analysis is for informational purposes only."
                
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Multimodal processing error: {str(e)}", exc_info=True)
            return ("I encountered an error processing your document. "
                   "For urgent matters, please consult a healthcare provider directly.")

    def _get_emergency_response(self) -> str:
        """Standard emergency response"""
        return """🚨 EMERGENCY MEDICAL ALERT 🚨

Your symptoms require immediate attention:
1. CALL EMERGENCY SERVICES NOW (911/112)
2. Do not ignore these symptoms
3. If alone, contact someone who can help
4. Stay on the line with emergency operator

Common emergency symptoms:
- Feeling like you might die
- Chest pain
- Severe difficulty breathing
- Sudden severe pain
- Loss of consciousness

This is not an AI response - this is standard emergency protocol."""