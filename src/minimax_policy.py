import numpy as np
from typing import Any, Dict, List, Optional, Union

import torch
import gymnasium as gym
from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, to_torch, to_torch_as

# (Keep existing imports)

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
        assert player_id in [1, 2], "player_id must be 1 or 2"
        self.action_space = action_space
        self.max_depth = max_depth
        self.player_id = player_id  # 1 or 2
        self.opponent_id = 3 - player_id  # If player is 1, opponent is 2 and vice versa
        self.eps = 0.0  # Epsilon for exploration (Minimax is deterministic, but kept for compatibility)
        self.max_action_num = action_space.n  # Number of possible actions (columns = 7)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MinimaxPolicy Init] Player ID: {self.player_id}, Opponent ID: {self.opponent_id}, Max Depth: {self.max_depth}")


    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs
    ) -> Batch:
        """Compute action over the given batch data with minimax algorithm."""
        obs = batch[input]
        actions = []
        all_logits = []

        for i in range(len(obs)):
            single_obs_data = obs[i]

            # Handle dict observation and extract mask
            action_mask = None
            if isinstance(single_obs_data, dict):
                if 'mask' in single_obs_data:
                    action_mask = single_obs_data['mask']
                if 'obs' in single_obs_data:
                    single_obs = single_obs_data['obs']
                else:
                    # Fallback if 'obs' key is missing but it's a dict
                    print("Warning: Observation is a dict but missing 'obs' key.")
                    single_obs = single_obs_data # Or handle error appropriately
            else:
                single_obs = single_obs_data

            # Convert to numpy if it's a torch tensor
            if isinstance(single_obs, torch.Tensor):
                single_obs = single_obs.cpu().numpy()

            board = self._obs_to_board(single_obs)

            # Use action mask if available, otherwise derive from board
            if action_mask is not None:
                valid_actions = [col for col, is_valid in enumerate(action_mask) if is_valid]
            else:
                valid_actions = self._get_valid_actions(board)


            if not valid_actions:
                 # This case should ideally be handled by the environment ending the game
                 print("Warning: No valid actions found for MinimaxPolicy. Choosing action 0.")
                 best_action = 0
            else:
                 best_action = self._get_best_action(board, valid_actions)
            actions.append(best_action)

            # Create dummy logits: high value for chosen action, low for others
            logits = np.full(self.max_action_num, -1e9) # Use very small value for non-chosen
            if best_action < self.max_action_num:
                 logits[best_action] = 1.0 # Set chosen action logit
            all_logits.append(logits)


        act = np.array(actions)
        logits_np = np.array(all_logits)
        logits_tensor = to_torch(logits_np, device=self.device)

        mask = getattr(batch.obs, "mask", None)
        if mask is not None:
            # Ensure mask has the correct shape (batch_size, num_actions)
            if len(mask.shape) == 1 and len(obs) == 1: # Handle single env case
                 mask = mask.reshape(1, -1)
            if mask.shape == logits_tensor.shape:
                 logits_tensor = self.compute_q_value(logits_tensor, mask)
            else:
                 print(f"Warning: Logits shape {logits_tensor.shape} and mask shape {mask.shape} mismatch. Skipping mask application on logits.")


        return Batch(logits=logits_tensor, act=act, state=None)


    def _obs_to_board(self, obs: np.ndarray) -> np.ndarray:
        """Convert observation (6, 7, 2) to board state (6, 7)."""
        if not isinstance(obs, np.ndarray) or obs.shape != (6, 7, 2):
             print(f"Warning: Invalid observation shape for _obs_to_board: {getattr(obs, 'shape', 'unknown')}. Returning empty board.")
             return np.zeros((6, 7), dtype=np.int8)

        board = np.zeros((6, 7), dtype=np.int8)
        # Channel 0: Current player's pieces (always map to self.player_id)
        # Channel 1: Opponent's pieces (always map to self.opponent_id)
        board[obs[:, :, 0] == 1] = self.player_id
        board[obs[:, :, 1] == 1] = self.opponent_id
        return board


    def _get_best_action(self, board: np.ndarray, valid_actions: List[int]) -> int:
        """Find the best action using win/block checks and minimax."""

        if not valid_actions:
            print("Error: _get_best_action called with no valid actions!")
            return 0

        # 1. Check if WE can win immediately
        for action in valid_actions:
            new_board = board.copy()
            if self._apply_action(new_board, action, self.player_id):
                if self._check_winner(new_board) == self.player_id:
                    print(f"Found winning move at column {action}")
                    return action

        # 2. Check if OPPONENT can win immediately, and block
        for action in valid_actions:
            new_board = board.copy()
            # Simulate the OPPONENT playing in the 'action' column
            if self._apply_action(new_board, action, self.opponent_id):
                 winner = self._check_winner(new_board)
                 if winner == self.opponent_id:
                     print(f"Blocking opponent's winning move at column {action}")
                     return action # Play 'action' to block

        # 3. If no immediate win/loss, use minimax
        best_value = float('-inf')
        # Ensure there's a default action in case all moves lead to negative infinity
        best_action = valid_actions[0]

        alpha = float('-inf')
        beta = float('inf')

        for action in valid_actions:
            new_board = board.copy()
            if self._apply_action(new_board, action, self.player_id):
                value = self._minimax(new_board, self.max_depth - 1, alpha, beta, False) # False: Minimizing player's turn

                if value > best_value:
                    best_value = value
                    best_action = action

                alpha = max(alpha, value)
                
        return best_action


    def _print_board(self, board: np.ndarray) -> None:
        """Print the board state for debugging."""
        symbols = {0: '.', self.player_id: 'X', self.opponent_id: 'O'} # Map 1/2 to X/O based on policy's perspective
        print("-" * (7 * 2))
        for r in range(6):
            row_str = ""
            for c in range(7):
                row_str += symbols.get(board[r, c], '?') + " " # Use get for safety
            print(row_str)
        print("0 1 2 3 4 5 6") # Column numbers
        print("-" * (7 * 2))


    def _minimax(self, board: np.ndarray, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        """Minimax algorithm with alpha-beta pruning."""
        winner = self._check_winner(board)
        valid_actions = self._get_valid_actions(board) # Get valid actions for the current state

        # Terminal conditions
        if winner == self.player_id:
            return 1000.0 + depth # Prioritize faster wins
        elif winner == self.opponent_id:
            return -1000.0 - depth # Penalize faster losses more
        elif winner == 0: # Draw
            return 0.0
        elif not valid_actions:
             print("Warning: Reached non-winning state with no valid moves in minimax.")
             return 0.0
        elif depth == 0:
            return self._evaluate_board(board) # Evaluate heuristic at max depth

        if is_maximizing: # Current player (self.player_id) wants to maximize score
            value = float('-inf')
            for action in valid_actions:
                new_board = board.copy()
                if self._apply_action(new_board, action, self.player_id):
                    value = max(value, self._minimax(new_board, depth - 1, alpha, beta, False)) # Next turn is minimizing
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break  # Beta cutoff
            return value
        else: # Opponent (self.opponent_id) wants to minimize score
            value = float('inf')
            for action in valid_actions:
                new_board = board.copy()
                if self._apply_action(new_board, action, self.opponent_id):
                    value = min(value, self._minimax(new_board, depth - 1, alpha, beta, True)) # Next turn is maximizing
                    beta = min(beta, value)
                    if alpha >= beta:
                        break  # Alpha cutoff
            return value


    def _get_valid_actions(self, board: np.ndarray) -> List[int]:
        """Get valid actions (indices of columns that are not full)."""
        return [col for col in range(7) if board[0, col] == 0]


    def _apply_action(self, board: np.ndarray, action: int, player: int) -> bool:
        """Apply an action (drop piece in column) to the board. Returns False if column is full."""
        if board[0, action] != 0:
             print(f"Warning: Attempted to play in full column {action}. Board state:")
             return False # Indicate failure: column is full
        for row in range(5, -1, -1):
            if board[row, action] == 0:
                board[row, action] = player
                return True # Indicate success
        return False


    def _check_winner(self, board: np.ndarray) -> int:
        """Check if there's a winner or draw. Returns player ID (1 or 2), 0 for draw, -1 for ongoing."""
        rows, cols = 6, 7
        # Check horizontal wins
        for r in range(rows):
            for c in range(cols - 3):
                if board[r, c] != 0 and board[r, c] == board[r, c + 1] == board[r, c + 2] == board[r, c + 3]:
                    return board[r, c]
        # Check vertical wins
        for r in range(rows - 3):
            for c in range(cols):
                if board[r, c] != 0 and board[r, c] == board[r + 1, c] == board[r + 2, c] == board[r + 3, c]:
                    return board[r, c]
        # Check positive diagonal wins (/)
        for r in range(3, rows):
            for c in range(cols - 3):
                if board[r, c] != 0 and board[r, c] == board[r - 1, c + 1] == board[r - 2, c + 2] == board[r - 3, c + 3]:
                    return board[r, c]
        # Check negative diagonal wins (\)
        for r in range(rows - 3):
            for c in range(cols - 3):
                if board[r, c] != 0 and board[r, c] == board[r + 1, c + 1] == board[r + 2, c + 2] == board[r + 3, c + 3]:
                    return board[r, c]

        # Check for draw (board full) *after* checking wins
        if np.all(board != 0):
            return 0  # Draw

        # No winner and not a draw
        return -1


    def _evaluate_board(self, board: np.ndarray) -> float:
        """Heuristic evaluation of a non-terminal board state."""
        score = 0.0
        rows, cols = 6, 7

        # --- Evaluate Windows of 4 ---
        # Horizontal
        for r in range(rows):
            for c in range(cols - 3):
                window = board[r, c:c+4]
                score += self._evaluate_window(window)
        # Vertical
        for r in range(rows - 3):
            for c in range(cols):
                window = board[r:r+4, c]
                score += self._evaluate_window(window)
        # Positive Diagonal (/)
        for r in range(3, rows):
            for c in range(cols - 3):
                window = np.array([board[r-i, c+i] for i in range(4)])
                score += self._evaluate_window(window)
        # Negative Diagonal (\)
        for r in range(rows - 3):
            for c in range(cols - 3):
                window = np.array([board[r+i, c+i] for i in range(4)])
                score += self._evaluate_window(window)

        # --- Center Column Preference ---
        center_col_idx = cols // 2
        center_array = board[:, center_col_idx]
        score += np.sum(center_array == self.player_id) * 3 # Give bonus for pieces in center

        return score

    def _evaluate_window(self, window: np.ndarray) -> float:
        """Evaluate a 4-cell window."""
        player_count = np.sum(window == self.player_id)
        opponent_count = np.sum(window == self.opponent_id)
        empty_count = 4 - player_count - opponent_count # More robust count

        # If mixed pieces, window is useless for winning line
        if player_count > 0 and opponent_count > 0:
            return 0.0

        # Score based on player's potential
        if player_count == 4: return 1000.0 # Should be caught by check_winner but safety
        if player_count == 3 and empty_count == 1: return 10.0 # Strong position
        if player_count == 2 and empty_count == 2: return 3.0  # Moderate position
        if player_count == 1 and empty_count == 3: return 1.0  # Weak position

        # Score based on opponent's potential (negative values)
        if opponent_count == 4: return -1000.0 # Should be caught by check_winner
        if opponent_count == 3 and empty_count == 1: return -50.0 # Try to block win
        if opponent_count == 2 and empty_count == 2: return -4.0  # Moderate threat
        if opponent_count == 1 and empty_count == 3: return -1.0  # Weak threat

        return 0.0 # All empty or cannot form a line

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Minimax does not learn from experience, so just return the batch."""
        return batch

    def learn(self, batch: Batch, **kwargs) -> Dict[str, float]:
        """Minimax does not learn."""
        return {} # Return empty dict as required

    def set_eps(self, eps: float) -> None:
        """Minimax is deterministic, but method needed for API compatibility."""
        self.eps = eps # Store eps value, though not used in core logic

    def train(self, mode: bool = True) -> "MinimaxPolicy":
        """Set training mode (no effect on Minimax)."""
        self.training = mode
        return self

    def eval(self, mode: bool = True) -> "MinimaxPolicy":
        """Set evaluation mode (no effect on Minimax)."""
        self.training = not mode
        return self

    def compute_q_value(
        self, logits: torch.Tensor, mask: Optional[np.ndarray]
    ) -> torch.Tensor:
        """Apply action mask to logits (used in Tianshou's collector)."""
        if mask is not None:
            # Ensure mask is torch tensor on the correct device
            mask_tensor = to_torch_as(mask, logits)
             # Use large negative number where mask is 0 (invalid action)
            min_value = torch.finfo(logits.dtype).min
            # Add large negative value to invalid actions
            # Ensure mask is broadcastable (e.g., (batch, actions))
            if mask_tensor.shape != logits.shape:
                 print(f"Warning: compute_q_value mask shape {mask_tensor.shape} != logits shape {logits.shape}")
                 # Attempt to broadcast if possible, otherwise return original logits
                 try:
                     logits = logits + (1 - mask_tensor) * min_value
                 except RuntimeError:
                     print("Error: Could not broadcast mask to logits. Returning unmasked logits.")
                     return logits
            else:
                 logits = logits + (1 - mask_tensor) * min_value

        return logits

    def exploration_noise(
        self,
        act: Union[np.ndarray, Batch],
        batch: Batch,
    ) -> Union[np.ndarray, Batch]:
        """ No exploration noise for deterministic Minimax. Method needed for API."""
        # Minimax is deterministic, exploration is not typically applied.
        # If exploration were desired (e.g., for variety), random choice
        # among top-k moves or epsilon-random could be added here.
        # For now, return the deterministic action.
        return act