This system combines classical AI (Minimax + Alpha-Beta Pruning) with modern deep reinforcement learning, fully accelerated with GPU-powered PyTorch.

🚀 What T3-DUEL Does

T3-DUEL is an autonomous game-learning system trained over 1000+ episodes, where:

A Dueling DDQN agent plays both X and O.

A Minimax agent provides optimal gameplay as an opponent.

The system uses state symmetries (rotations/flips) to multiply training data.

GPU acceleration significantly speeds up Q-learning.

The model learns optimal decision-making without hardcoded strategies.

This setup forces the neural agent to learn near-perfect gameplay — even when playing against a perfect algorithmic opponent.

🧠 Tech Used

PyTorch (GPU-accelerated)

Dueling DDQN architecture

Experience Replay Buffer (20,000 samples)

Minimax with Alpha-Beta pruning

State symmetry augmentation

OpenCV for visualizing each game

NumPy + Python for environment simulation

🎯 What Makes It Special

✔ Learns both offensive and defensive strategies
✔ Combines classical and neural AI — a hybrid approach
✔ Uses advanced concepts like:

Value & Advantage Streams

Target Networks

Epsilon-Greedy Exploration

Game-state normalization

✔ Continuously improves by playing against a perfect Minimax engine

📈 Why I Built This

To understand how reinforcement learning agents learn optimal policy when facing an unbeatable opponent — and to explore hybrid AI systems that blend symbolic reasoning and neural learning.

Projects like this improve your intuition for:

Deep Q-learning

Game theory

Model stability

Environment-agent dynamics

🤖 System Name: T3-DUEL

T3 = Tic-Tac-Toe
DUEL = Dueling DDQN + Minimax training loop

A compact, memorable name for a hybrid learning engine.
