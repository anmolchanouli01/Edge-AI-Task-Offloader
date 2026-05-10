import random
import os
import torch
from universal_env import UniversalOffloadEnv
from universal_agent import UniversalAgent

def main():
    env = UniversalOffloadEnv()
    
    # Enable Adapt Mode! We want it to learn the new hardware specs.
    agent = UniversalAgent(env.state_size, env.action_size, adapt_mode=True)
    batch_size = 64
    
    # Try to load existing brain if you transferred it to this new machine
    model_path = "universal_model.pth"
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"✅ Loaded base brain from {model_path}. Starting Continuous Adaptation...")
    else:
        print("⚠️ No base brain found. Starting fresh training on this hardware...")

    print("\n🚀 System is LIVE. Routing tasks and fine-tuning to local hardware limits.")
    
    task_count = 0
    try:
        # Infinite loop for a real-time running system
        while True:
            task_count += 1
            task_complexity = random.randint(1, 10) 
            
            # Observe -> Act -> Evaluate -> Learn
            state = env.get_state(task_complexity)
            action = agent.act(state)
            latency, reward = env.execute_task(action, task_complexity)
            next_state = env.get_state(task_complexity)
            
            # The agent continues to remember and train in the background!
            agent.remember(state, action, reward, next_state)
            agent.replay(batch_size)
            
            if task_count % 10 == 0:
                action_str = "🖥️ LOCAL" if action == 0 else "☁️ CLOUD"
                print(f"Task {task_count} | Epsilon: {agent.epsilon:.2f} | "
                      f"Level: {task_complexity:2d} | CPU: {state[0]:4.1f}% | Bat: {state[2]:3.0f}% | "
                      f"Routed to: {action_str} | Latency: {latency:.3f}s")
                      
    except KeyboardInterrupt:
        # When you stop the script, it saves its new, adapted brain
        print("\n🛑 System stopped. Saving adapted hardware profile...")
        agent.save_model("universal_model.pth")
        print("💾 New brain saved.")

if __name__ == "__main__":
    main()