"""
AI Safety Pipeline - Proof of Concept
Integrates abuse detection, escalation tracking, crisis intervention, and age filtering
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
from collections import defaultdict, deque

class AISafetyPipeline:
    def __init__(self, offensive_model_path, suicide_model_path):
        # Load models
        self.offensive_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.offensive_model = AutoModelForSequenceClassification.from_pretrained(offensive_model_path)
        
        self.suicide_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.suicide_model = AutoModelForSequenceClassification.from_pretrained(suicide_model_path)
        
        # Class mappings
        self.offensive_classes = {0: 'hate_speech', 1: 'offensive_language', 2: 'neither'}
        self.suicide_classes = {0: 'non-suicide', 1: 'suicide'}
        
        # Conversation history for escalation tracking
        self.user_history = defaultdict(lambda: deque(maxlen=10))
        self.escalation_scores = defaultdict(float)
        
        # Age-inappropriate keywords
        self.adult_keywords = ['sex', 'drug', 'alcohol', 'violence', 'gambling']
        
    def abuse_check(self, text):
        """Step 1: Check for abusive content"""
        inputs = self.offensive_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.offensive_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        return {
            'class': self.offensive_classes[predicted_class],
            'confidence': confidence,
            'is_abusive': predicted_class in [0, 1]
        }
    
    def escalation_check(self, user_id, abuse_result):
        """Step 2: Check for escalation patterns"""
        self.user_history[user_id].append({
            'timestamp': datetime.now(),
            'is_abusive': abuse_result['is_abusive'],
            'class': abuse_result['class']
        })
        
        recent_messages = list(self.user_history[user_id])[-5:]
        abusive_count = sum(1 for msg in recent_messages if msg['is_abusive'])
        escalation_score = abusive_count / len(recent_messages) if recent_messages else 0
        self.escalation_scores[user_id] = escalation_score
        
        return {
            'escalation_score': escalation_score,
            'is_escalating': escalation_score > 0.6,
            'recent_abusive_count': abusive_count
        }
    
    def crisis_check(self, text):
        """Step 3: Check for crisis/suicide indicators"""
        inputs = self.suicide_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.suicide_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        return {
            'class': self.suicide_classes[predicted_class],
            'confidence': confidence,
            'is_crisis': predicted_class == 1
        }
    
    def age_filter(self, text, user_age=None):
        """Step 4: Age-appropriate content filtering"""
        if user_age is None:
            return {'age_appropriate': True, 'reason': 'No age specified'}
        
        text_lower = text.lower()
        inappropriate_found = [keyword for keyword in self.adult_keywords if keyword in text_lower]
        
        if user_age < 13 and inappropriate_found:
            return {'age_appropriate': False, 'reason': f'Contains: {inappropriate_found}'}
        elif user_age < 18 and any(word in text_lower for word in ['sex', 'drug']):
            return {'age_appropriate': False, 'reason': 'Adult content detected'}
        
        return {'age_appropriate': True, 'reason': 'Content appropriate'}
    
    def process_message(self, text, user_id, user_age=None):
        """Main pipeline: Process message through all safety checks"""
        result = {
            'message': text,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'safety_checks': {}
        }
        
        # Step 1: Abuse Check
        abuse_result = self.abuse_check(text)
        result['safety_checks']['abuse'] = abuse_result
        
        # Step 2: Escalation Check
        escalation_result = self.escalation_check(user_id, abuse_result)
        result['safety_checks']['escalation'] = escalation_result
        
        # Step 3: Crisis Check
        crisis_result = self.crisis_check(text)
        result['safety_checks']['crisis'] = crisis_result
        
        # Step 4: Age Filter
        age_result = self.age_filter(text, user_age)
        result['safety_checks']['age_filter'] = age_result
        
        # Generate final decision
        result['decision'] = self._make_decision(abuse_result, escalation_result, crisis_result, age_result)
        
        return result
    
    def _make_decision(self, abuse_result, escalation_result, crisis_result, age_result):
        """Generate final safety decision and alerts"""
        alerts = []
        action = "ALLOW"
        
        # Crisis takes highest priority
        if crisis_result['is_crisis']:
            alerts.append("CRISIS_INTERVENTION_REQUIRED")
            action = "BLOCK_AND_ALERT"
        
        # High escalation
        elif escalation_result['is_escalating']:
            alerts.append("ESCALATION_DETECTED")
            action = "WARN_AND_MONITOR"
        
        # Abuse detection
        elif abuse_result['is_abusive']:
            if abuse_result['class'] == 'hate_speech':
                alerts.append("HATE_SPEECH_DETECTED")
                action = "BLOCK"
            else:
                alerts.append("OFFENSIVE_LANGUAGE_DETECTED")
                action = "WARN"
        
        # Age filtering
        if not age_result['age_appropriate']:
            alerts.append("AGE_INAPPROPRIATE_CONTENT")
            if action == "ALLOW":
                action = "BLOCK"
        
        return {
            'action': action,
            'alerts': alerts,
            'requires_human_review': "CRISIS_INTERVENTION_REQUIRED" in alerts or "ESCALATION_DETECTED" in alerts
        }

def main():
    """Demo the AI Safety Pipeline"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build dynamic paths relative to script location
    offensive_model_path = os.path.join(script_dir, "Offensive_Detection", "Output", "checkpoint-840")
    suicide_model_path = os.path.join(script_dir, "Suicide_Detection", "Output", "checkpoint-1965")
    
    pipeline = AISafetyPipeline(
        offensive_model_path=offensive_model_path,
        suicide_model_path=suicide_model_path
    )
    
    test_messages = [
        {"text": "Hello, how are you today?", "user_id": "user1", "age": 25},
        {"text": "I hate this stupid system", "user_id": "user1", "age": 25},
        {"text": "You're such an idiot", "user_id": "user1", "age": 25},
        {"text": "I want to end it all", "user_id": "user2", "age": 20},
        {"text": "Let's talk about drugs and alcohol", "user_id": "user3", "age": 12},
    ]
    
    print("=== AI Safety Pipeline Demo ===\n")
    
    for i, msg in enumerate(test_messages, 1):
        print(f"Message {i}: {msg['text']}")
        result = pipeline.process_message(msg['text'], msg['user_id'], msg['age'])
        
        print(f"Decision: {result['decision']['action']}")
        if result['decision']['alerts']:
            print(f"Alerts: {', '.join(result['decision']['alerts'])}")
        print(f"Human Review Required: {result['decision']['requires_human_review']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
