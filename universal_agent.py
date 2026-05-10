import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Input is now 5
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class UniversalAgent:
    def __init__(self, state_size, action_size, adapt_mode=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        
        self.gamma = 0.95    
        
        # If adapting to a new system, don't start totally random (1.0), 
        # start at 15% exploration so it relies mostly on past knowledge.
        self.epsilon = 0.15 if adapt_mode else 1.0   
        self.epsilon_min = 0.05 if adapt_mode else 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0005 
        
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor([t[1] for t in minibatch]).unsqueeze(1)
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        
        current_q_values = self.model(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            max_next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values)

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
        
    def save_model(self, file_name="universal_model.pth"):
        torch.save(self.model.state_dict(), file_name)