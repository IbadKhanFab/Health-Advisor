import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
from typing import Dict, Any
import google.generativeai as genai
import certifi

load_dotenv()

class HealthChatbot:
    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        if not test_mode:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
            self.mongo_uri = os.getenv("MONGO_URI")
            self.db_name = os.getenv("MONGO_DB_NAME", "health_chatbot")

            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            if not self.mongo_uri:
                raise ValueError("MONGO_URI not found in environment variables")

            # Initialize Gemini and MongoDB
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash-latest") 

            self.client = MongoClient(self.mongo_uri, tlsCAFile=certifi.where())
            self.db = self.client[self.db_name]
            self.conversations = self.db["conversations"]

    def analyze_health_input(self, user_input: str, file_content: str = None) -> Dict[str, Any]:
        try:
            prompt = self._build_prompt(user_input, file_content)

            if self.test_mode:
                response = self._get_test_response(user_input, file_content)
            else:
                response = self._call_gemini_api(prompt)

            conversation_id = self._store_conversation(
                user_input=user_input,
                file_content=file_content,
                ai_response=response
            )

            return {
                "status": "success",
                "response": response,
                "conversation_id": conversation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            if any(word in user_input.lower() for word in ["die", "dying", "heart attack", "stroke"]):
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

    def _build_prompt(self, symptoms: str, file_content: str = None) -> str:
        prompt = f"""Patient reports: "{symptoms}"\n\n"""
        if file_content:
            prompt += f"""Test results provided:\n{file_content}\n\n"""
        prompt += """Please provide:
- Possible conditions (ranked by likelihood)
- Immediate recommended actions
- When to seek emergency care
- Follow-up recommendations
- Standard medical disclaimer"""
        return prompt

    def _call_gemini_api(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip() if response and response.text else "No response from Gemini"
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")

    def _get_test_response(self, symptoms: str, file_content: str = None) -> str:
        if "die" in symptoms.lower() or "dying" in symptoms.lower():
            return f"""EMERGENCY RESPONSE:
- Your symptoms suggest a potential medical emergency
- Please call emergency services immediately (911/112)
- Do not delay seeking help
- If alone, contact someone who can assist you

TEST RESULTS NOTE: {file_content or 'No test results provided'}"""

        return f"""TEST MODE RESPONSE:
For symptoms: "{symptoms}"
Possible conditions: [Example Condition 1, Example Condition 2]
Recommended actions: 
- Consult with a healthcare provider
- Monitor symptoms
- Seek emergency care if symptoms worsen

Test results: {file_content or 'No test results provided'}"""

    def _get_emergency_response(self) -> str:
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

    def _store_conversation(self, user_input: str, ai_response: str, file_content: str = None) -> str:
        conversation = {
            "user_input": user_input,
            "file_content": file_content,
            "ai_response": ai_response,
            "timestamp": datetime.utcnow()
        }
        result = self.conversations.insert_one(conversation)
        return str(result.inserted_id)


if __name__ == "__main__":
    print("Health Chatbot (type 'quit' to exit)")

    chatbot = HealthChatbot(test_mode=False)  
    while True:
        try:
            symptoms = input("\nDescribe your symptoms: ").strip()
            if symptoms.lower() == 'quit':
                break

            file_content = None
            if input("Do you have a test file to upload? (y/n): ").lower() == 'y':
                file_content = "Simulated test results:\n- Heart rate: 120 bpm\n- BP: 150/95\n- Oxygen: 92%"
                print("File content included in analysis")

            print("\nAnalyzing your input...")
            result = chatbot.analyze_health_input(symptoms, file_content)

            print("\n=== Health Recommendation ===")
            print(result["response"])
            if result.get("conversation_id"):
                print(f"\nReference ID: {result['conversation_id']}")

        except Exception as e:
            print(f"\nError: {str(e)}\nPlease try again or contact support.")
