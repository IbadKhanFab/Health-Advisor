import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "health_chatbot")
    
    EMERGENCY_PHRASES = [
        r"\b die\b", r"\b dying\b", r"\bheart attack\b", r"\b stroke\b", 
        r"\b can't breathe\b", r"\b cannot breathe\b", r"\b suicide\b", 
        r"\b kill myself\b", r"\b emergency\b", r"\b911\b", r"\b ambulance\b"
    ]
    
    MEDICAL_KEYWORDS = ["blood", "test","Medicines", "result", "count", "level", "mg", "dl", "mmol"]
    
    @staticmethod
    def validate():
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        if not Config.MONGO_URI:
            raise ValueError("MONGO_URI not found in environment variables")