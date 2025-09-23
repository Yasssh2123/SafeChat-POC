"""
CLI Demo for AI Safety Pipeline
Command-line interface for testing the safety system
"""

from ai_safety_pipeline import AISafetyPipeline
import json

def print_result(result):
    """Pretty print the analysis result"""
    print(f"\n{'='*60}")
    print(f"MESSAGE: {result['message']}")
    print(f"USER: {result['user_id']}")
    print(f"{'='*60}")
    
    # Decision
    decision = result['decision']
    print(f"\n🎯 DECISION: {decision['action']}")
    
    if decision['alerts']:
        print(f"🚨 ALERTS: {', '.join(decision['alerts'])}")
    
    if decision['requires_human_review']:
        print("⚠️  HUMAN REVIEW REQUIRED")
    
    # Safety checks
    checks = result['safety_checks']
    
    print(f"\n📊 DETAILED ANALYSIS:")
    print(f"  Abuse: {checks['abuse']['class']} (conf: {checks['abuse']['confidence']:.3f})")
    print(f"  Crisis: {checks['crisis']['class']} (conf: {checks['crisis']['confidence']:.3f})")
    print(f"  Escalation: {checks['escalation']['escalation_score']:.3f}")
    print(f"  Age Filter: {'✅' if checks['age_filter']['age_appropriate'] else '❌'}")

def main():
    print("🛡️  AI Safety Pipeline - CLI Demo")
    print("Loading models...")
    
    # Initialize pipeline
    pipeline = AISafetyPipeline(
        offensive_model_path="C:/Users/yashc/Downloads/Solulab/Offensive_Detection/Output/checkpoint-840",
        suicide_model_path="C:/Users/yashc/Downloads/Solulab/Suicide_Detection/Output/checkpoint-1965"
    )
    
    print("✅ Models loaded successfully!\n")
    
    # Interactive mode
    print("Enter messages to analyze (type 'quit' to exit, 'demo' for sample tests):")
    
    user_id = "cli_user"
    user_age = 25
    
    while True:
        try:
            message = input("\n> ").strip()
            
            if message.lower() == 'quit':
                break
            elif message.lower() == 'demo':
                run_demo(pipeline)
                continue
            elif not message:
                continue
            
            # Process message
            result = pipeline.process_message(message, user_id, user_age)
            print_result(result)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def run_demo(pipeline):
    """Run predefined demo messages"""
    demo_messages = [
        {"text": "Hello, how are you today?", "user_id": "demo1", "age": 25},
        {"text": "I hate this stupid system", "user_id": "demo1", "age": 25},
        {"text": "You're such an idiot", "user_id": "demo1", "age": 25},
        {"text": "I want to end it all", "user_id": "demo2", "age": 20},
        {"text": "Let's talk about drugs and alcohol", "user_id": "demo3", "age": 12},
    ]
    
    print("\n🎯 Running Demo Messages...")
    
    for msg in demo_messages:
        result = pipeline.process_message(msg['text'], msg['user_id'], msg['age'])
        print_result(result)
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()