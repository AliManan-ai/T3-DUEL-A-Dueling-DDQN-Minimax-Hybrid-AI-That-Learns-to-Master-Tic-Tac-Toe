import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import random
from game import TicTacToe, DuelingQNetwork, draw_board

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for mouse callback
human_move = None

def mouse_callback(event, x, y, flags, param):
    global human_move
    if event == cv2.EVENT_LBUTTONDOWN:
        # Map x, y to 0-8
        col = x // 100
        row = y // 100
        if 0 <= col < 3 and 0 <= row < 3:
            human_move = row * 3 + col

def main():
    global human_move
    
    # Load model
    model = DuelingQNetwork().to(device)
    try:
        model.load_state_dict(torch.load("tictactoe_dueling_ddqn.pth"))
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model file not found. Please run game.py first to train the model.")
        return

    env = TicTacToe()
    env.reset()
    done = False
    
    # Randomly assign roles
    if random.random() < 0.5:
        human_player = 1
        ai_player = 2
        print("Game Start! You are Player 1 (X). AI is Player 2 (O).")
    else:
        human_player = 2
        ai_player = 1
        print("Game Start! AI is Player 1 (X). You are Player 2 (O).")

    print("Click on the board to make a move.")
    
    cv2.namedWindow("Tic Tac Toe - Human vs AI")
    cv2.setMouseCallback("Tic Tac Toe - Human vs AI", mouse_callback)
    
    turn = 1 # Player 1 starts
    
    while not done:
        img = draw_board(env.board)
        cv2.imshow("Tic Tac Toe - Human vs AI", img)
        key = cv2.waitKey(10)
        if key == 27: # ESC to quit
            break
        
        available_actions = env.get_available_actions()
        
        if turn == ai_player: # AI turn
            # Small delay for better UX
            time.sleep(0.5)
            
            # Get state from AI's perspective
            ai_state = env.get_state(ai_player)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(ai_state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                
                # Mask invalid actions
                mask = torch.full((1, 9), -float('inf')).to(device)
                mask[0, available_actions] = 0
                masked_q_values = q_values + mask
                
                action = masked_q_values.argmax().item()
            
            print(f"AI chose position {action}")
            reward, done = env.step(action, ai_player)
            turn = human_player
            
        else: # Human turn
            if human_move is not None:
                if human_move in available_actions:
                    reward, done = env.step(human_move, human_player)
                    turn = ai_player
                    human_move = None # Reset for next turn
                else:
                    print("Invalid move or spot taken.")
                    human_move = None
            
    img = draw_board(env.board)
    cv2.imshow("Tic Tac Toe - Human vs AI", img)
    
    if env.winner == human_player:
        print("Congratulations! You won!")
        cv2.putText(img, "You Won!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    elif env.winner == ai_player:
        print("AI won! Better luck next time.")
        cv2.putText(img, "AI Won!", (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    else:
        print("It's a draw!")
        cv2.putText(img, "Draw!", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        
    cv2.imshow("Tic Tac Toe - Human vs AI", img)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
