import numpy as np
from typing import Any, Dict, List, Optional, Union

import torch
import gymnasium as gym
from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, to_torch, to_torch_as


class MinimaxPolicy(BasePolicy):
    """Minimax policy with alpha-beta pruning for Connect 4.
    
    This policy implements the minimax algorithm with alpha-beta pruning
    to find the best move in Connect 4. It uses a heuristic evaluation
    function to evaluate non-terminal board states.
    """
    
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        max_depth: int = 4,
        player_id: int = 1,  # 1 or 2
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.action_space = action_space
        self.max_depth = max_depth
        self.player_id = player_id  # 1 or 2
        self.opponent_id = 3 - player_id  # If player is 1, opponent is 2 and vice versa
        self.eps = 0.0  # Epsilon for exploration
        self.max_action_num = action_space.n  # Number of possible actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs
    ) -> Batch:
        """Compute action over the given batch data with minimax algorithm.
            
        Args:
            batch: A batch of observations.
            state: Not used in minimax.
            model: Not used in minimax.
            input: The key in batch to use as the input observation.
            
        Returns:
            A Batch with the following keys:
            * ``act`` the action.
            * ``logits`` the network's raw output (dummy for minimax).
            * ``state`` the hidden state (None for minimax).
        """
        # Get the observation from the batch
        obs = batch[input]
        
        # For each observation in the batch, find the best action using minimax
        actions = []
        
        mask = None
        if hasattr(batch.obs, "mask"):
            mask = batch.obs.mask
            # print("Action mask found:", mask)
        
        for i in range(len(obs)):
            # Get the single observation
            single_obs = obs[i]
            
            if isinstance(single_obs, torch.Tensor):
                single_obs = single_obs.cpu().numpy()
            
            # Convert observation to board state
            board = self._obs_to_board(single_obs)
            
            # Find the best action using minimax
            best_action = self._get_best_action(board)
            
            # If there's an action mask, make sure the selected action is legal
            if mask is not None:
                # Get the mask for this observation
                obs_mask = mask[i] if i < len(mask) else mask
                
                # If the selected action is illegal (mask is 0), choose a legal action instead
                if obs_mask[best_action] == 0:
                    print(f"WARNING: Action {best_action} is illegal according to the mask!")
                    print("Action mask:", obs_mask)
                    
                    # Find legal actions (where mask is 1)
                    legal_actions = np.where(obs_mask == 1)[0]
                    if len(legal_actions) > 0:
                        # Choose a legal action
                        best_action = legal_actions[0]
                        print(f"Choosing legal action {best_action} instead.")
            
            actions.append(best_action)
        
        # Create dummy logits for compatibility with Tianshou
        # For each action, create a tensor with a high value for the chosen action
        # and low values for other actions
        act = np.array(actions)
        bsz = len(act)
        logits = np.zeros((bsz, self.max_action_num))
        
        # Set a high value (10.0) for the chosen action and low values (0.0) for others
        for i, a in enumerate(act):
            logits[i, a] = 10.0
        
        # Convert to torch tensor
        logits_tensor = to_torch(logits, device=self.device)
        
        # Apply action mask if provided
        mask = getattr(batch.obs, "mask", None)
        if mask is not None:
            # Apply the mask to the logits
            logits_tensor = self.compute_q_value(logits_tensor, mask)
            
            # Also apply the mask to the actions
            # This ensures that we never select an illegal action
            for i in range(len(act)):
                # Get the mask for this observation
                obs_mask = mask[i] if len(mask) > i else mask
                
                # If the selected action is illegal (mask is 0), choose a legal action instead
                if obs_mask[act[i]] == 0:
                    # Find legal actions (where mask is 1)
                    legal_actions = np.where(obs_mask == 1)[0]
                    if len(legal_actions) > 0:
                        # Choose the legal action with the highest logit value
                        legal_logits = logits[i][legal_actions]
                        best_legal_idx = np.argmax(legal_logits)
                        act[i] = legal_actions[best_legal_idx]
                        # Update the logits to reflect the new action
                        logits[i] = np.zeros_like(logits[i])
                        logits[i][act[i]] = 10.0
            
            # Update the logits tensor with the new logits
            logits_tensor = to_torch(logits, device=self.device)
        
        return Batch(logits=logits_tensor, act=act, state=None)
    
    def _obs_to_board(self, obs: Union[np.ndarray, dict, 'Batch']) -> np.ndarray:
        """Convert observation to board state.
        
        The observation from Connect 4 environment is a 2D array with shape (6, 7, 2)
        where the first channel represents player 1's pieces and the second channel
        represents player 2's pieces. We convert it to a single 2D array where
        0 represents empty, 1 represents player 1's pieces, and 2 represents player 2's pieces.
        
        Args:
            obs: Observation from the environment.
            
        Returns:
            A 2D array representing the board state.
        """
        # Connect 4 board is 6x7
        board = np.zeros((6, 7), dtype=np.int8)
        
        # Handle Batch object from Tianshou
        if hasattr(obs, '__class__') and obs.__class__.__name__ == 'Batch':
            # Try to get the 'obs' attribute from the Batch
            if hasattr(obs, 'obs'):
                obs = obs.obs
                if isinstance(obs, dict) and 'obs' in obs:
                    obs = obs['obs']
            else:
                # Try to access the Batch as a dictionary
                try:
                    if 'obs' in obs:
                        obs = obs['obs']
                except:
                    pass
        
        # Handle dictionary
        if isinstance(obs, dict) and 'obs' in obs:
            obs = obs['obs']
        
        # Handle empty observation
        if obs is None or (hasattr(obs, 'shape') and len(obs.shape) == 0):
            # Return an empty board if the observation is empty
            return board
        
        # Extract player and opponent pieces from observation
        # The exact format depends on the environment, this is based on common formats
        try:
            print(f"Observation shape: {getattr(obs, 'shape', 'unknown')}")
            
            if hasattr(obs, 'shape') and obs.shape == (6, 7, 2):  # Assuming shape is (height, width, channels)
                player_pieces = obs[:, :, 0]
                opponent_pieces = obs[:, :, 1]
                
                # Set player pieces to player_id
                board[player_pieces == 1] = self.player_id
                
                # Set opponent pieces to opponent_id
                board[opponent_pieces == 1] = self.opponent_id
                
                print(f"Player pieces count: {np.sum(player_pieces == 1)}")
                print(f"Opponent pieces count: {np.sum(opponent_pieces == 1)}")
                
            elif hasattr(obs, 'shape') and obs.shape == (2, 6, 7):  # Assuming shape is (channels, height, width)
                player_pieces = obs[0]
                opponent_pieces = obs[1]
                
                # Set player pieces to player_id
                board[player_pieces == 1] = self.player_id
                
                # Set opponent pieces to opponent_id
                board[opponent_pieces == 1] = self.opponent_id
                
                print(f"Player pieces count: {np.sum(player_pieces == 1)}")
                print(f"Opponent pieces count: {np.sum(opponent_pieces == 1)}")
                
            elif hasattr(obs, 'shape') and len(obs.shape) == 3:
                # Try to infer the format based on the shape
                print(f"Trying to infer format from shape: {obs.shape}")
                
                if obs.shape[0] == 2:  # Channels first
                    player_pieces = obs[0]
                    opponent_pieces = obs[1]
                elif obs.shape[2] == 2:  # Channels last
                    player_pieces = obs[:, :, 0]
                    opponent_pieces = obs[:, :, 1]
                else:
                    print(f"Cannot infer format from shape: {obs.shape}")
                    return board
                
                # Set player pieces to player_id
                board[player_pieces == 1] = self.player_id
                
                # Set opponent pieces to opponent_id
                board[opponent_pieces == 1] = self.opponent_id
                
                print(f"Player pieces count: {np.sum(player_pieces == 1)}")
                print(f"Opponent pieces count: {np.sum(opponent_pieces == 1)}")
                
            else:
                # If the observation format is different, log a warning and return an empty board
                print(f"Warning: Unsupported observation shape: {getattr(obs, 'shape', 'unknown')}. Using empty board.")
                if hasattr(obs, 'shape'):
                    print(f"Observation content sample: {obs.flatten()[:10] if hasattr(obs, 'flatten') else 'cannot flatten'}")
        except Exception as e:
            # If there's an error processing the observation, log a warning and return an empty board
            print(f"Warning: Error processing observation: {e}. Using empty board.")
        
        return board
    
    def _get_best_action(self, board: np.ndarray) -> int:
        """Find the best action using minimax with alpha-beta pruning.
        
        Args:
            board: Current board state.
            
        Returns:
            The best action to take.
        """
        print("\n=== MINIMAX DECISION MAKING ===")
        print(f"Player ID: {self.player_id}, Opponent ID: {self.opponent_id}")
        print("Current board state:")
        self._print_board(board)
        
        valid_actions = self._get_valid_actions(board)
        print(f"Valid actions: {valid_actions}")
        
        if not valid_actions:
            print("No valid actions, returning random action")
            # If no valid actions, return a random action (should not happen in normal play)
            return self.action_space.sample()
        
        # First, check if we can win in one move
        for action in valid_actions:
            new_board = board.copy()
            self._apply_action(new_board, action, self.player_id)
            if self._check_winner(new_board) == self.player_id:
                print(f"Found winning move at column {action}")
                return action
        
        # Then, check if opponent has a winning move and block it
        for action in valid_actions:
            # Check if this action would be a winning move for the opponent
            new_board = board.copy()
            self._apply_action(new_board, action, self.opponent_id)
            winner = self._check_winner(new_board)
            if winner == self.opponent_id:
                print(f"Blocking opponent's winning move at column {action}")
                print("Board if opponent plays here:")
                self._print_board(new_board)
                return action
        
        print("No immediate threats, proceeding with minimax")
        
        # If no immediate threat, proceed with minimax
        best_value = float('-inf')
        best_action = valid_actions[0]
        
        # Try each valid action and choose the one with the highest value
        for action in valid_actions:
            # Make a copy of the board and apply the action
            new_board = board.copy()
            self._apply_action(new_board, action, self.player_id)
            
            # Get the value of the resulting board state
            value = self._minimax(new_board, self.max_depth - 1, float('-inf'), float('inf'), False)
            print(f"Action {action} has value {value}")
            
            # Update best action if this action has a higher value
            if value > best_value:
                best_value = value
                best_action = action
        
        print(f"Choosing action {best_action} with value {best_value}")
        return best_action
        
    def _print_board(self, board: np.ndarray) -> None:
        """Print the board state for debugging.
        
        Args:
            board: Current board state.
        """
        symbols = {0: '.', 1: 'X', 2: 'O'}
        for row in range(6):
            row_str = ""
            for col in range(7):
                row_str += symbols[board[row, col]] + " "
            print(row_str)
        print("0 1 2 3 4 5 6")  # Column numbers
    
    def _minimax(self, board: np.ndarray, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        """Minimax algorithm with alpha-beta pruning.
        
        Args:
            board: Current board state.
            depth: Current depth in the search tree.
            alpha: Alpha value for pruning.
            beta: Beta value for pruning.
            is_maximizing: Whether this is a maximizing node.
            
        Returns:
            The value of the board state.
        """
        # Check if the game is over or we've reached the maximum depth
        winner = self._check_winner(board)
        if winner == self.player_id:
            return 1000  # Player wins
        elif winner == self.opponent_id:
            return -1000  # Opponent wins
        elif winner == 0:  # Draw
            return 0
        elif depth == 0:
            return self._evaluate_board(board)
        
        valid_actions = self._get_valid_actions(board)
        if not valid_actions:
            return 0  # Draw
        
        if is_maximizing:
            value = float('-inf')
            for action in valid_actions:
                new_board = board.copy()
                self._apply_action(new_board, action, self.player_id)
                value = max(value, self._minimax(new_board, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # Beta cutoff
            return value
        else:
            value = float('inf')
            for action in valid_actions:
                new_board = board.copy()
                self._apply_action(new_board, action, self.opponent_id)
                value = min(value, self._minimax(new_board, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if alpha >= beta:
                    break  # Alpha cutoff
            return value
    
    def _get_valid_actions(self, board: np.ndarray) -> List[int]:
        """Get valid actions for the current board state.
        
        Args:
            board: Current board state.
            
        Returns:
            List of valid actions (column indices).
        """
        valid_actions = []
        for col in range(7):
            if board[0, col] == 0:  # If the top cell in the column is empty
                valid_actions.append(col)
        return valid_actions
    
    def _apply_action(self, board: np.ndarray, action: int, player: int) -> None:
        """Apply an action to the board.
        
        Args:
            board: Current board state.
            action: Action to apply (column index).
            player: Player making the move (1 or 2).
        """
        # Find the lowest empty row in the selected column
        for row in range(5, -1, -1):
            if board[row, action] == 0:
                board[row, action] = player
                break
    
    def _check_winner(self, board: np.ndarray) -> int:
        """Check if there's a winner on the board.
        
        Args:
            board: Current board state.
            
        Returns:
            0 if no winner or draw, 1 if player 1 wins, 2 if player 2 wins.
        """
        # Check horizontal
        for row in range(6):
            for col in range(4):
                if (board[row, col] != 0 and
                    board[row, col] == board[row, col + 1] == 
                    board[row, col + 2] == board[row, col + 3]):
                    return board[row, col]
        
        # Check vertical
        for row in range(3):
            for col in range(7):
                if (board[row, col] != 0 and
                    board[row, col] == board[row + 1, col] == 
                    board[row + 2, col] == board[row + 3, col]):
                    return board[row, col]
        
        # Check diagonal (positive slope)
        for row in range(3, 6):
            for col in range(4):
                if (board[row, col] != 0 and
                    board[row, col] == board[row - 1, col + 1] == 
                    board[row - 2, col + 2] == board[row - 3, col + 3]):
                    return board[row, col]
        
        # Check diagonal (negative slope)
        for row in range(3):
            for col in range(4):
                if (board[row, col] != 0 and
                    board[row, col] == board[row + 1, col + 1] == 
                    board[row + 2, col + 2] == board[row + 3, col + 3]):
                    return board[row, col]
        
        # Check if the board is full (draw)
        if np.all(board[0, :] != 0):
            return 0  # Draw
        
        # No winner yet
        return -1
    
    def _evaluate_board(self, board: np.ndarray) -> float:
        """Heuristic evaluation function for non-terminal board states.
        
        This function evaluates the board state based on the number of potential
        winning lines for each player. A potential winning line is a line of 4
        cells that could potentially form a winning line (i.e., it doesn't contain
        any opponent's pieces).
        
        Args:
            board: Current board state.
            
        Returns:
            A score representing how good the board state is for the player.
            Positive values are good for the player, negative values are good
            for the opponent.
        """
        score = 0
        
        # Check horizontal windows
        for row in range(6):
            for col in range(4):
                window = board[row, col:col+4]
                score += self._evaluate_window(window)
        
        # Check vertical windows
        for row in range(3):
            for col in range(7):
                window = board[row:row+4, col]
                score += self._evaluate_window(window)
        
        # Check diagonal windows (positive slope)
        for row in range(3, 6):
            for col in range(4):
                window = [board[row-i, col+i] for i in range(4)]
                score += self._evaluate_window(window)
        
        # Check diagonal windows (negative slope)
        for row in range(3):
            for col in range(4):
                window = [board[row+i, col+i] for i in range(4)]
                score += self._evaluate_window(window)
        
        # Prefer center column
        center_col = board[:, 3]
        score += np.sum(center_col == self.player_id) * 3
        
        return score
    
    def _evaluate_window(self, window: np.ndarray) -> float:
        """Evaluate a window of 4 cells.
        
        Args:
            window: A window of 4 cells.
            
        Returns:
            A score for the window.
        """
        player_count = np.sum(window == self.player_id)
        opponent_count = np.sum(window == self.opponent_id)
        empty_count = np.sum(window == 0)
        
        # If the window contains both player and opponent pieces, it's not a potential win
        if player_count > 0 and opponent_count > 0:
            return 0
        
        # Score based on the number of player pieces in the window
        if player_count == 4:
            return 100  # Player wins
        elif player_count == 3 and empty_count == 1:
            return 5  # Player has 3 in a row with an empty cell
        elif player_count == 2 and empty_count == 2:
            return 2  # Player has 2 in a row with 2 empty cells
        elif player_count == 1 and empty_count == 3:
            return 1  # Player has 1 in a row with 3 empty cells
        
        # Score based on the number of opponent pieces in the window
        if opponent_count == 4:
            return -100  # Opponent wins
        elif opponent_count == 3 and empty_count == 1:
            return -50  # Opponent has 3 in a row with an empty cell - high priority to block!
        elif opponent_count == 2 and empty_count == 2:
            return -2  # Opponent has 2 in a row with 2 empty cells
        elif opponent_count == 1 and empty_count == 3:
            return -1  # Opponent has 1 in a row with 3 empty cells
        
        return 0
    
    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Process the batch data before learning.
        
        Since MinimaxPolicy doesn't actually learn, this function is a simple pass-through.
        It's implemented for compatibility with the Tianshou framework.
        
        Args:
            batch: A batch of data.
            buffer: Replay buffer.
            indices: Indices of the batch data in the replay buffer.
            
        Returns:
            The processed batch data.
        """
        return batch
    
    def learn(self, batch: Batch, **kwargs) -> Dict[str, Any]:
        """Dummy learn function (minimax doesn't learn).
        
        Args:
            batch: A batch of data.
            
        Returns:
            An empty dict.
        """
        return {}
    
    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps
        
    def train(self, mode: bool = True) -> "MinimaxPolicy":
        """Set the module in training mode."""
        self.training = mode
        return self
        
    def compute_q_value(
        self, logits: torch.Tensor, mask: Optional[np.ndarray]
    ) -> torch.Tensor:
        """Compute the q value based on the network's raw output and action mask."""
        if mask is not None:
            # the masked q value should be smaller than logits.min()
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_torch_as(1 - mask, logits) * min_value
        return logits
        
    def exploration_noise(
        self,
        act: Union[np.ndarray, Batch],
        batch: Batch,
    ) -> Union[np.ndarray, Batch]:
        """Add exploration noise to the action."""
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self.max_action_num)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act
