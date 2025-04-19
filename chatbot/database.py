import os
from pymongo import MongoClient
import certifi
from bson import ObjectId
from datetime import datetime
from typing import List, Dict, Optional
from .config import Config
import logging

logger = logging.getLogger("HealthChatbot")

class Database:
    def __init__(self):
        Config.validate()
        self.client = MongoClient(Config.MONGO_URI, tlsCAFile=certifi.where())
        self.db = self.client[Config.MONGO_DB_NAME]
        self.conversations = self.db["conversations"]
        self.chats = self.db["chats"]
    
    def start_new_chat(self, person_id: str) -> str:
        """Start a new chat session for a person"""
        try:
            chat_data = {
                "person_id": person_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "status": "active",
                "conversation_ids": []
            }
            result = self.chats.insert_one(chat_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error starting new chat: {str(e)}", exc_info=True)
            raise RuntimeError("Could not start new chat session")

    def get_chat_history(self, chat_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history for a chat session"""
        try:
            chat = self.chats.find_one({"_id": ObjectId(chat_id)})
            if not chat:
                return []
            
            conversation_ids = chat.get("conversation_ids", [])
            conversations = list(self.conversations.find(
                {"_id": {"$in": conversation_ids}},
                {"user_input": 1, "ai_response": 1, "timestamp": 1, "_id": 0}
            ).sort("timestamp", -1).limit(limit))
            
            return conversations[::-1]  # Return in chronological order
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}", exc_info=True)
            return []

    def store_conversation(self, person_id: str, chat_id: str, user_input: str, 
                         ai_response: str, file_path: Optional[str] = None, 
                         file_content: Optional[str] = None) -> ObjectId:
        """Store conversation in MongoDB"""
        try:
            conversation = {
                "person_id": person_id,
                "chat_id": ObjectId(chat_id),
                "user_input": user_input,
                "file_path": os.path.basename(file_path) if file_path else None,
                "file_content_sample": file_content[:500] if file_content else None,
                "file_content_length": len(file_content) if file_content else 0,
                "ai_response": ai_response,
                "timestamp": datetime.utcnow()
            }
            result = self.conversations.insert_one(conversation)
            return result.inserted_id
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}", exc_info=True)
            raise RuntimeError("Could not store conversation")