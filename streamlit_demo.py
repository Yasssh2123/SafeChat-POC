"""
Streamlit Demo for AI Safety Pipeline
Interactive web interface to test the safety system
"""

import os
import streamlit as st
import pandas as pd
from ai_safety_pipeline import AISafetyPipeline
import json
from datetime import datetime

# Initialize pipeline (cached for performance)
@st.cache_resource
def load_pipeline():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    offensive_model_path = os.path.join(script_dir, "Offensive_Detection", "Output", "checkpoint-840")
    suicide_model_path = os.path.join(script_dir, "Suicide_Detection", "Output", "checkpoint-1965")
    
    return AISafetyPipeline(
        offensive_model_path=offensive_model_path,
        suicide_model_path=suicide_model_path
    )

def main():
    st.set_page_config(page_title="AI Safety Pipeline Demo", page_icon="🛡️", layout="wide")
    
    st.title("🛡️ AI Safety Pipeline - Proof of Concept")
    st.markdown("**Real-time content safety analysis using ML models**")
    
    # Load pipeline
    pipeline = load_pipeline()
    
    # Sidebar for user settings
    st.sidebar.header("User Settings")
    user_id = st.sidebar.text_input("User ID", value="demo_user")
    user_age = st.sidebar.number_input("User Age", min_value=1, max_value=100, value=25)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Message Safety Check")
        
        # Message input
        message = st.text_area("Enter message to analyze:", height=100, 
                              placeholder="Type your message here...")
        
        if st.button("🔍 Analyze Message", type="primary"):
            if message.strip():
                with st.spinner("Analyzing message..."):
                    result = pipeline.process_message(message, user_id, user_age)
                
                # Display results
                st.subheader("Analysis Results")
                
                # Decision summary
                decision = result['decision']
                action_color = {
                    "ALLOW": "green",
                    "WARN": "orange", 
                    "WARN_AND_MONITOR": "orange",
                    "BLOCK": "red",
                    "BLOCK_AND_ALERT": "red"
                }
                
                st.markdown(f"**Action:** :{action_color.get(decision['action'], 'gray')}[{decision['action']}]")
                
                if decision['alerts']:
                    st.warning(f"🚨 Alerts: {', '.join(decision['alerts'])}")
                
                if decision['requires_human_review']:
                    st.error("⚠️ **Human review required**")
                
                # Detailed results
                with st.expander("Detailed Analysis", expanded=True):
                    checks = result['safety_checks']
                    
                    # Abuse Check
                    st.markdown("**1. Abuse Detection**")
                    abuse = checks['abuse']
                    st.write(f"- Classification: {abuse['class']}")
                    st.write(f"- Confidence: {abuse['confidence']:.3f}")
                    st.write(f"- Is Abusive: {abuse['is_abusive']}")
                    
                    # Escalation Check
                    st.markdown("**2. Escalation Pattern**")
                    escalation = checks['escalation']
                    st.write(f"- Escalation Score: {escalation['escalation_score']:.3f}")
                    st.write(f"- Is Escalating: {escalation['is_escalating']}")
                    st.write(f"- Recent Abusive Messages: {escalation['recent_abusive_count']}")
                    
                    # Crisis Check
                    st.markdown("**3. Crisis Detection**")
                    crisis = checks['crisis']
                    st.write(f"- Classification: {crisis['class']}")
                    st.write(f"- Confidence: {crisis['confidence']:.3f}")
                    st.write(f"- Crisis Detected: {crisis['is_crisis']}")
                    
                    # Age Filter
                    st.markdown("**4. Age Appropriateness**")
                    age_filter = checks['age_filter']
                    st.write(f"- Age Appropriate: {age_filter['age_appropriate']}")
                    st.write(f"- Reason: {age_filter['reason']}")
            else:
                st.warning("Please enter a message to analyze")
    
    with col2:
        st.header("User History")
        
        if user_id in pipeline.user_history:
            history = list(pipeline.user_history[user_id])
            escalation_score = pipeline.escalation_scores.get(user_id, 0)
            
            st.metric("Current Escalation Score", f"{escalation_score:.3f}")
            
            if history:
                st.subheader("Recent Messages")
                for i, msg in enumerate(reversed(history[-5:]), 1):
                    status = "🔴" if msg['is_abusive'] else "🟢"
                    st.write(f"{status} {msg['class']} ({msg['timestamp'].strftime('%H:%M:%S')})")
        else:
            st.info("No message history for this user")
    


if __name__ == "__main__":
    main()
