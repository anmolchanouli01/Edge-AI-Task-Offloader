import psutil
import time
import random

class UniversalOffloadEnv:
    def __init__(self):
        # NEW STATE: [CPU_%, RAM_%, Battery_%, Is_Plugged_In (1/0), Task_Complexity]
        self.state_size = 5
        self.action_size = 2 # 0: Local, 1: Cloud

    def get_state(self, task_complexity):
        cpu = psutil.cpu_percent(interval=0.05)
        ram = psutil.virtual_memory().percent
        
        # Cross-platform battery check
        battery = psutil.sensors_battery()
        if battery is not None:
            battery_percent = battery.percent
            is_plugged_in = 1 if battery.power_plugged else 0
        else:
            # If it's a desktop without a battery, assume infinite power
            battery_percent = 100.0
            is_plugged_in = 1
            
        return [cpu, ram, battery_percent, is_plugged_in, task_complexity]

    def execute_task(self, action, task_complexity):
        start_time = time.time()
        
        # Get live battery info for rewards
        battery = psutil.sensors_battery()
        is_plugged_in = battery.power_plugged if battery else True
        battery_percent = battery.percent if battery else 100
        
        if action == 0:
            # ACTION 0: LOCAL EXECUTION
            current_cpu = psutil.cpu_percent(interval=None) / 100.0
            simulated_processing_time = (task_complexity * 0.01) * (1 + current_cpu)
            time.sleep(simulated_processing_time) 
            
            latency = time.time() - start_time
            
            # Hardware Penalties
            ram_penalty = -100 if psutil.virtual_memory().percent > 85 else 0
            
            # NEW: Severe penalty for draining a dying, unplugged battery
            energy_penalty = 0
            if battery_percent < 20 and not is_plugged_in:
                energy_penalty = -200 
                
            reward = -latency + ram_penalty + energy_penalty
            
        else:
            # ACTION 1: CLOUD EXECUTION
            network_delay = random.uniform(0.01, 0.05) 
            cloud_processing = task_complexity * 0.002
            time.sleep(network_delay + cloud_processing)
            
            latency = time.time() - start_time
            cloud_cost_penalty = -1.5 
            reward = -latency + cloud_cost_penalty
            
        return latency, reward