import numpy as np
import cv2
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import collections

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        return self.board

    def get_state(self, player_id):
        # Return flattened board from perspective of player_id
        # Self = 1, Opponent = -1, Empty = 0
        state = np.zeros(9, dtype=float)
        flat_board = self.board.reshape(9)
        
        for i in range(9):
            if flat_board[i] == player_id:
                state[i] = 1.0
            elif flat_board[i] == 0:
                state[i] = 0.0
            else:
                state[i] = -1.0
        return state

    def get_available_actions(self):
        return [i for i, x in enumerate(self.board.reshape(9)) if x == 0]

    def step(self, action, player):
        if self.done:
            return 0, self.done

        row = action // 3
        col = action % 3

        if self.board[row, col] != 0:
            # Invalid move
            return -10, self.done

        self.board[row, col] = player

        if self.check_winner(player):
            self.done = True
            self.winner = player
            return 10, self.done
        
        if len(self.get_available_actions()) == 0:
            self.done = True
            self.winner = 0 # Draw
            return 0, self.done

        return 0, self.done

    def check_winner(self, player):
        # Check rows and columns
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        # Check diagonals
        if self.board[0, 0] == player and self.board[1, 1] == player and self.board[2, 2] == player:
            return True
        if self.board[0, 2] == player and self.board[1, 1] == player and self.board[2, 0] == player:
            return True
        return False
    
    def copy(self):
        new_env = TicTacToe()
        new_env.board = self.board.copy()
        new_env.done = self.done
        new_env.winner = self.winner
        return new_env

class MinimaxAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        self.opponent_id = 3 - player_id
        self.memo = {}

    def get_action(self, env):
        # Minimax with Alpha-Beta Pruning
        # We need to clear memo if we switch sides or just use a key that includes player_id?
        # The memo key currently is (board_str + is_maximizing). 
        # is_maximizing is relative to self.player_id.
        # So if we reuse this instance for different player_ids, we might need to be careful.
        # But we instantiate a new MinimaxAgent or update player_id.
        # Let's just clear memo if we change player_id or include it in key.
        # For safety, let's just clear memo at start of game or use a robust key.
        # Given the small state space, we can just keep it.
        _, action = self.minimax(env, True)
        return action

    def minimax(self, env, is_maximizing):
        # Key must include whose turn it is or perspective
        state_key = str(env.board.reshape(9)) + str(is_maximizing) + str(self.player_id)
        if state_key in self.memo:
            return self.memo[state_key]

        if env.check_winner(self.player_id):
            return 1, None
        if env.check_winner(self.opponent_id):
            return -1, None
        available_actions = env.get_available_actions()
        if not available_actions:
            return 0, None

        if is_maximizing:
            best_score = -float('inf')
            best_action = None
            for action in available_actions:
                new_env = env.copy()
                new_env.step(action, self.player_id)
                score, _ = self.minimax(new_env, False)
                if score > best_score:
                    best_score = score
                    best_action = action
            self.memo[state_key] = (best_score, best_action)
            return best_score, best_action
        else:
            best_score = float('inf')
            best_action = None
            for action in available_actions:
                new_env = env.copy()
                new_env.step(action, self.opponent_id)
                score, _ = self.minimax(new_env, True)
                if score < best_score:
                    best_score = score
                    best_action = action
            self.memo[state_key] = (best_score, best_action)
            return best_score, best_action

class DuelingQNetwork(nn.Module):
    def __init__(self):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(9, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Value stream
        self.value_fc = nn.Linear(256, 128)
        self.value = nn.Linear(128, 1)
        
        # Advantage stream
        self.advantage_fc = nn.Linear(256, 128)
        self.advantage = nn.Linear(128, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        val = torch.relu(self.value_fc(x))
        val = self.value(val)
        
        adv = torch.relu(self.advantage_fc(x))
        adv = self.advantage(adv)
        
        return val + adv - adv.mean(dim=1, keepdim=True)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

def get_symmetries(state, action, next_state):
    # state: 9 elements array
    # action: int 0-8
    # next_state: 9 elements array
    
    # Convert to 3x3
    state_3x3 = state.reshape(3, 3)
    next_state_3x3 = next_state.reshape(3, 3)
    
    # Action mask (one-hot) to rotate action
    action_mask = np.zeros((3, 3))
    action_mask[action // 3, action % 3] = 1
    
    symmetries = []
    
    # 4 rotations
    for k in range(4):
        s_rot = np.rot90(state_3x3, k)
        ns_rot = np.rot90(next_state_3x3, k)
        a_rot = np.rot90(action_mask, k)
        
        symmetries.append((s_rot.reshape(9), np.argmax(a_rot), ns_rot.reshape(9)))
        
        # Flip horizontal
        s_flip = np.fliplr(s_rot)
        ns_flip = np.fliplr(ns_rot)
        a_flip = np.fliplr(a_rot)
        
        symmetries.append((s_flip.reshape(9), np.argmax(a_flip), ns_flip.reshape(9)))
        
    return symmetries

class DDQNAgent:
    def __init__(self, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        self.policy_net = DuelingQNetwork().to(device)
        self.target_net = DuelingQNetwork().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
        self.memory = ReplayBuffer(20000) # Larger buffer for symmetries
        self.batch_size = 64
        self.gamma = 0.99
        
        self.last_state = None
        self.last_action = None

    def get_action(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            
            mask = torch.full((1, 9), -float('inf')).to(device)
            mask[0, available_actions] = 0
            masked_q_values = q_values + mask
            
            return masked_q_values.argmax().item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        state = torch.FloatTensor(np.array(state)).to(device)
        action = torch.LongTensor(action).unsqueeze(1).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        q_values = self.policy_net(state).gather(1, action)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_state).argmax(1).unsqueeze(1)
            next_q_values = self.target_net(next_state).gather(1, next_actions)
            expected_q_values = reward + (self.gamma * next_q_values * (1 - done))
        
        loss = nn.MSELoss()(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        # Add symmetries
        symmetries = get_symmetries(state, action, next_state)
        for s, a, ns in symmetries:
            self.memory.push(s, a, reward, ns, done)

def draw_board(board):
    img = np.zeros((300, 300, 3), dtype=np.uint8) + 255 
    
    cv2.line(img, (100, 0), (100, 300), (0, 0, 0), 2)
    cv2.line(img, (200, 0), (200, 300), (0, 0, 0), 2)
    cv2.line(img, (0, 100), (300, 100), (0, 0, 0), 2)
    cv2.line(img, (0, 200), (300, 200), (0, 0, 0), 2)

    for i in range(3):
        for j in range(3):
            if board[i, j] == 1: 
                center = (j * 100 + 50, i * 100 + 50)
                cv2.line(img, (center[0]-30, center[1]-30), (center[0]+30, center[1]+30), (0, 0, 255), 3)
                cv2.line(img, (center[0]+30, center[1]-30), (center[0]-30, center[1]+30), (0, 0, 255), 3)
            elif board[i, j] == 2: 
                center = (j * 100 + 50, i * 100 + 50)
                cv2.circle(img, center, 30, (255, 0, 0), 3)
    
    return img

def main():
    env = TicTacToe()
    agent = DDQNAgent() # One agent learns both roles
    
    episodes = 1000
    target_update = 50
    
    print("Starting training: Dueling DDQN vs Minimax (Both Roles + Symmetries)...")
    
    for episode in range(episodes):
        env.reset()
        done = False
        
        # Randomize role: 0 = Agent is P1 (X), 1 = Agent is P2 (O)
        agent_role = random.choice([1, 2])
        minimax_role = 3 - agent_role
        
        # Initialize Minimax with correct role
        minimax_agent = MinimaxAgent(player_id=minimax_role)
        
        agent.last_state = None
        agent.last_action = None
        
        turn = 1 # Player 1 starts
        
        while not done:
            if (episode + 1) % 100 == 0:
                img = draw_board(env.board)
                cv2.imshow("Training", img)
                cv2.waitKey(1)
            
            available_actions = env.get_available_actions()
            
            if turn == agent_role:
                # Agent Turn
                state = env.get_state(agent_role)
                action = agent.get_action(state, available_actions)
                
                # Delayed reward update from previous turn
                if agent.last_state is not None:
                    agent.remember(agent.last_state, agent.last_action, 0, state, False)
                    agent.update()

                reward, done = env.step(action, agent_role)
                
                if done:
                    if env.winner == agent_role:
                        agent.remember(state, action, 10, np.zeros(9), True)
                    elif env.winner == 0:
                        agent.remember(state, action, 0, np.zeros(9), True)
                    agent.update()
                else:
                    agent.last_state = state
                    agent.last_action = action
                
            else:
                # Minimax Turn
                action = minimax_agent.get_action(env)
                
                reward, done = env.step(action, minimax_role)
                
                if done:
                    if env.winner == minimax_role:
                        # Agent lost
                        if agent.last_state is not None:
                            agent.remember(agent.last_state, agent.last_action, -10, env.get_state(agent_role), True)
                            agent.update()
                    elif env.winner == 0:
                        # Draw
                        if agent.last_state is not None:
                            agent.remember(agent.last_state, agent.last_action, 0, env.get_state(agent_role), True)
                            agent.update()
                else:
                    # Game continues, Agent's move was safe
                    pass

            turn = 3 - turn # Switch turn (1 -> 2, 2 -> 1)
        
        agent.update_epsilon()
        
        if episode % target_update == 0:
            agent.update_target_network()

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} completed. Epsilon: {agent.epsilon:.2f}")

    # Save the model
    torch.save(agent.policy_net.state_dict(), "tictactoe_dueling_ddqn.pth")
    print("Model saved to tictactoe_dueling_ddqn.pth")

    cv2.destroyAllWindows()
    print("Training finished.")

if __name__ == "__main__":
    main()
