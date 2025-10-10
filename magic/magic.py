"""
MAGIC (Model and Guided Importance Sampling Combining) Estimator
for Offline Policy Evaluation

Implementation of the MAGIC algorithm for evaluating policies using
offline data with model-based and importance sampling techniques.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Union
from copy import deepcopy
import cvxpy as cp
from scipy.stats import bootstrap
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.policy_models import MLP, DiagGaussian, ActorProb


@dataclass
class Trajectory:
    """Represents a single trajectory/episode"""
    states: np.ndarray  # (T, state_dim)
    actions: np.ndarray  # (T, action_dim)
    rewards: np.ndarray  # (T,)
    next_states: np.ndarray  # (T, state_dim)
    terminals: np.ndarray  # (T,)

    def __len__(self):
        return len(self.states)


class TrajectoryReconstructor:
    """Reconstructs trajectories from raw transition data"""

    def __init__(self,
                 distance_threshold: float = 100.0,
                 min_episode_length: int = 5):
        """
        Args:
            distance_threshold: State distance threshold for detecting episode boundaries
            min_episode_length: Minimum length before checking for episode reset
        """
        self.distance_threshold = distance_threshold
        self.min_episode_length = min_episode_length

    def is_terminal(self, terminal_flag: float, next_state: np.ndarray) -> bool:
        """
        Check if a state is terminal

        Args:
            terminal_flag: Terminal flag from data (1.0 if terminal, 0.0 otherwise)
            next_state: Next state

        Returns:
            True if state is terminal
        """
        # In the buffer, terminals are stored as 1.0 for terminal, 0.0 for non-terminal
        return terminal_flag > 0.5

    def is_new_episode(self,
                       state: np.ndarray,
                       next_state: np.ndarray,
                       step_count: int) -> bool:
        """
        Detect if transition represents start of new episode

        Args:
            state: Current state
            next_state: Next state
            step_count: Current step count in trajectory

        Returns:
            True if this is a new episode
        """
        if step_count < self.min_episode_length:
            return False

        # Check for large state jumps (indicates reset)
        distance = np.linalg.norm(next_state - state)
        if distance > self.distance_threshold:
            return True

        return False

    def reconstruct_trajectories(self, buffer) -> List[Trajectory]:
        """
        Reconstruct trajectories from replay buffer

        Args:
            buffer: ReplayBuffer containing transition data

        Returns:
            List of Trajectory objects
        """
        trajectories = []
        current_states = []
        current_actions = []
        current_rewards = []
        current_next_states = []
        current_terminals = []

        n_transitions = buffer.size

        for i in range(n_transitions):
            state = buffer.observations[i]
            action = buffer.actions[i]
            reward = buffer.rewards[i].item()
            next_state = buffer.next_observations[i]
            terminal = buffer.terminals[i].item()

            # Add transition to current trajectory
            current_states.append(state)
            current_actions.append(action)
            current_rewards.append(reward)
            current_next_states.append(next_state)
            current_terminals.append(terminal)

            # Check if episode ends
            is_terminal = self.is_terminal(terminal, next_state)
            is_new_ep = self.is_new_episode(state, next_state, len(current_states))

            if is_terminal or is_new_ep or i == n_transitions - 1:
                # Save trajectory if it has at least one transition
                if len(current_states) > 0:
                    trajectories.append(Trajectory(
                        states=np.array(current_states),
                        actions=np.array(current_actions),
                        rewards=np.array(current_rewards),
                        next_states=np.array(current_next_states),
                        terminals=np.array(current_terminals)
                    ))

                # Reset for next trajectory
                current_states = []
                current_actions = []
                current_rewards = []
                current_next_states = []
                current_terminals = []

        print(f"Reconstructed {len(trajectories)} trajectories from {n_transitions} transitions")
        print(f"Average trajectory length: {np.mean([len(t) for t in trajectories]):.2f}")

        return trajectories


class BehaviorPolicyLearner:
    """Learns behavior policy from offline data using behavior cloning"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 action_space,
                 hidden_dims: List[int] = [256, 256],
                 lr: float = 3e-4,
                 device: str = "cuda"):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_space: Gym action space object
            hidden_dims: Hidden layer dimensions
            lr: Learning rate
            device: Device to use
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.device = torch.device(device)

        # Build policy network (Gaussian policy for continuous actions)
        backbone = MLP(input_dim=state_dim, hidden_dims=hidden_dims)
        dist_net = DiagGaussian(
            latent_dim=getattr(backbone, "output_dim"),
            output_dim=action_dim,
            unbounded=True,
            conditioned_sigma=True
        )
        self.policy = ActorProb(backbone, dist_net, device=device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def learn(self,
              states: np.ndarray,
              actions: np.ndarray,
              batch_size: int = 256,
              epochs: int = 50,
              early_stop_threshold: float = 0.0005) -> Dict[str, float]:
        """
        Learn behavior policy using behavior cloning

        Args:
            states: State data (N, state_dim)
            actions: Action data (N, action_dim)
            batch_size: Batch size for training
            epochs: Number of training epochs
            early_stop_threshold: Early stopping threshold for loss

        Returns:
            Training statistics
        """
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)

        n_samples = len(states)
        n_batches = n_samples // batch_size

        prev_loss = np.inf
        losses = []

        print(f"Training behavior policy on {n_samples} samples...")

        for epoch in range(epochs):
            total_loss = 0
            indices = np.random.permutation(n_samples)

            for i in range(n_batches):
                batch_idx = indices[i * batch_size:(i + 1) * batch_size]
                state_batch = states[batch_idx]
                action_batch = actions[batch_idx]

                # Get policy distribution
                dist = self.policy.get_dist(state_batch)

                # Negative log-likelihood loss
                log_prob = dist.log_prob(action_batch)
                loss = -log_prob.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / n_batches
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

            # Early stopping
            if epoch > 0 and abs(avg_loss - prev_loss) < early_stop_threshold:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            prev_loss = avg_loss

        return {"final_loss": losses[-1], "epochs": len(losses)}

    def log_prob(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Compute log probability of actions under policy

        Args:
            states: States (N, state_dim)
            actions: Actions (N, action_dim)

        Returns:
            Log probabilities (N,)
        """
        self.policy.eval()
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)

            dist = self.policy.get_dist(states)
            log_probs = dist.log_prob(actions)

        return log_probs.cpu().numpy().flatten()

    def prob(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Compute probability of actions under policy

        Args:
            states: States (N, state_dim)
            actions: Actions (N, action_dim)

        Returns:
            Probabilities (N,)
        """
        return np.exp(self.log_prob(states, actions))


class OffPolicyEstimator:
    """Computes various off-policy return estimators"""

    def __init__(self, gamma: float = 0.99):
        """
        Args:
            gamma: Discount factor
        """
        self.gamma = gamma

    def compute_importance_weights(self,
                                   trajectories: List[Trajectory],
                                   eval_policy,
                                   behavior_policy,
                                   clip_max: float = 100.0) -> List[np.ndarray]:
        """
        Compute per-step importance weights for all trajectories

        Args:
            trajectories: List of trajectories
            eval_policy: Evaluation policy
            behavior_policy: Behavior policy
            clip_max: Maximum importance weight (for stability)

        Returns:
            List of importance weight arrays (one per trajectory)
        """
        all_weights = []

        for traj in trajectories:
            # Get log probabilities under both policies
            eval_log_probs = eval_policy.log_prob(traj.states, traj.actions)
            behavior_log_probs = behavior_policy.log_prob(traj.states, traj.actions)

            # Compute importance weights
            log_weights = eval_log_probs - behavior_log_probs
            weights = np.exp(log_weights)

            # Clip for stability
            weights = np.minimum(weights, clip_max)

            all_weights.append(weights)

        return all_weights

    def importance_sampling(self,
                           trajectories: List[Trajectory],
                           eval_policy,
                           behavior_policy) -> float:
        """
        Vanilla importance sampling (IS) estimator

        Args:
            trajectories: List of trajectories
            eval_policy: Evaluation policy
            behavior_policy: Behavior policy

        Returns:
            IS estimate of policy value
        """
        weights_list = self.compute_importance_weights(
            trajectories, eval_policy, behavior_policy
        )

        total_return = 0.0

        for traj, weights in zip(trajectories, weights_list):
            # Compute cumulative importance weight
            cumulative_weight = np.cumprod(weights)

            # Compute discounted return
            discounts = self.gamma ** np.arange(len(traj))
            weighted_rewards = cumulative_weight * discounts * traj.rewards

            total_return += np.sum(weighted_rewards)

        return total_return / len(trajectories)

    def weighted_importance_sampling(self,
                                    trajectories: List[Trajectory],
                                    eval_policy,
                                    behavior_policy) -> float:
        """
        Weighted importance sampling (WIS) estimator

        Args:
            trajectories: List of trajectories
            eval_policy: Evaluation policy
            behavior_policy: Behavior policy

        Returns:
            WIS estimate of policy value
        """
        weights_list = self.compute_importance_weights(
            trajectories, eval_policy, behavior_policy
        )

        weighted_return = 0.0
        total_weight = 0.0

        for traj, weights in zip(trajectories, weights_list):
            # Compute cumulative importance weight
            cumulative_weight = np.cumprod(weights)

            # Compute discounted return
            discounts = self.gamma ** np.arange(len(traj))
            traj_return = np.sum(discounts * traj.rewards)

            # Final weight for trajectory
            final_weight = cumulative_weight[-1]

            weighted_return += final_weight * traj_return
            total_weight += final_weight

        if total_weight == 0:
            return 0.0

        return weighted_return / total_weight

    def per_decision_importance_sampling(self,
                                        trajectories: List[Trajectory],
                                        eval_policy,
                                        behavior_policy) -> float:
        """
        Per-decision importance sampling (PDIS) estimator

        Args:
            trajectories: List of trajectories
            eval_policy: Evaluation policy
            behavior_policy: Behavior policy

        Returns:
            PDIS estimate of policy value
        """
        weights_list = self.compute_importance_weights(
            trajectories, eval_policy, behavior_policy
        )

        total_return = 0.0

        for traj, weights in zip(trajectories, weights_list):
            T = len(traj)

            for t in range(T):
                # Importance weight up to time t
                weight_t = np.prod(weights[:t+1])

                # Discounted reward at time t
                discounted_reward = (self.gamma ** t) * traj.rewards[t]

                total_return += weight_t * discounted_reward

        return total_return / len(trajectories)

    def model_based_return(self,
                          trajectories: List[Trajectory],
                          eval_policy,
                          world_model,
                          rollout_length: int = 5) -> float:
        """
        Model-based return estimator using learned dynamics

        Args:
            trajectories: List of trajectories
            eval_policy: Evaluation policy
            world_model: Learned dynamics model
            rollout_length: Length of model rollouts

        Returns:
            Model-based estimate of policy value
        """
        total_return = 0.0

        for traj in trajectories:
            # Sample initial state from trajectory
            init_state = traj.states[0]

            # Rollout using model
            state = init_state
            traj_return = 0.0

            for t in range(min(rollout_length, len(traj))):
                # Sample action from evaluation policy
                action = eval_policy.sample_action(
                    state.reshape(1, -1), deterministic=False
                ).flatten()

                # Predict next state and reward using model
                next_state, reward, terminal, _ = world_model.predict(
                    state, action, deterministic=False
                )

                # Add discounted reward
                traj_return += (self.gamma ** t) * reward.item()

                # Check termination
                if terminal.item() > 0.5:
                    break

                state = next_state.flatten()

            total_return += traj_return

        return total_return / len(trajectories)

    def doubly_robust(self,
                     trajectories: List[Trajectory],
                     eval_policy,
                     behavior_policy,
                     value_function) -> float:
        """
        Doubly robust (DR) estimator

        Args:
            trajectories: List of trajectories
            eval_policy: Evaluation policy
            behavior_policy: Behavior policy
            value_function: Learned value function V(s)

        Returns:
            DR estimate of policy value
        """
        weights_list = self.compute_importance_weights(
            trajectories, eval_policy, behavior_policy
        )

        total_return = 0.0

        for traj, weights in zip(trajectories, weights_list):
            T = len(traj)

            # Initial state value
            V_0 = value_function(traj.states[0].reshape(1, -1)).item()
            total_return += V_0

            # Corrections
            for t in range(T):
                weight_t = np.prod(weights[:t+1])
                discount_t = self.gamma ** t

                # Value correction
                V_t = value_function(traj.states[t].reshape(1, -1)).item()
                if t < T - 1:
                    V_next = value_function(traj.next_states[t].reshape(1, -1)).item()
                else:
                    V_next = 0.0

                # TD error
                td_error = traj.rewards[t] + self.gamma * V_next - V_t

                total_return += weight_t * discount_t * td_error

        return total_return / len(trajectories)

    def compute_j_step_return(self,
                             trajectories: List[Trajectory],
                             eval_policy,
                             behavior_policy,
                             world_model,
                             j: int) -> Tuple[float, List[float]]:
        """
        Compute j-step return estimator

        Args:
            trajectories: List of trajectories
            eval_policy: Evaluation policy
            behavior_policy: Behavior policy
            world_model: Learned dynamics model (can be None if j=-1)
            j: Return length (-1 for IS, 0 for model-based, 1+ for hybrid)

        Returns:
            Tuple of (mean estimate, per-trajectory returns)
        """
        per_traj_returns = []

        if j == -1:
            # Pure importance sampling
            weights_list = self.compute_importance_weights(
                trajectories, eval_policy, behavior_policy
            )

            for traj, weights in zip(trajectories, weights_list):
                cumulative_weight = np.cumprod(weights)
                discounts = self.gamma ** np.arange(len(traj))
                traj_return = np.sum(cumulative_weight * discounts * traj.rewards)
                per_traj_returns.append(traj_return)

        elif j == 0:
            # Pure model-based
            for traj in trajectories:
                init_state = traj.states[0]
                state = init_state
                traj_return = 0.0

                for t in range(len(traj)):
                    action = eval_policy.sample_action(
                        state.reshape(1, -1), deterministic=False
                    ).flatten()

                    next_state, reward, terminal, _ = world_model.predict(
                        state, action, deterministic=False
                    )

                    traj_return += (self.gamma ** t) * reward.item()

                    if terminal.item() > 0.5:
                        break

                    state = next_state.flatten()

                per_traj_returns.append(traj_return)

        else:
            # Hybrid: j steps of IS, then model rollout
            weights_list = self.compute_importance_weights(
                trajectories, eval_policy, behavior_policy
            )

            for traj, weights in zip(trajectories, weights_list):
                traj_return = 0.0

                # First j steps with importance sampling
                for t in range(min(j, len(traj))):
                    weight_t = np.prod(weights[:t+1])
                    discount_t = self.gamma ** t
                    traj_return += weight_t * discount_t * traj.rewards[t]

                # Remaining steps with model
                if j < len(traj):
                    state = traj.next_states[j-1]
                    cumulative_weight = np.prod(weights[:j])

                    for t in range(j, len(traj)):
                        action = eval_policy.sample_action(
                            state.reshape(1, -1), deterministic=False
                        ).flatten()

                        next_state, reward, terminal, _ = world_model.predict(
                            state, action, deterministic=False
                        )

                        traj_return += cumulative_weight * (self.gamma ** t) * reward.item()

                        if terminal.item() > 0.5:
                            break

                        state = next_state.flatten()

                per_traj_returns.append(traj_return)

        return np.mean(per_traj_returns), per_traj_returns


class MAGICEstimator:
    """
    MAGIC (Model and Guided Importance Sampling Combining) Estimator
    for offline policy evaluation
    """

    def __init__(self, gamma: float = 0.99):
        """
        Args:
            gamma: Discount factor
        """
        self.gamma = gamma
        self.estimator = OffPolicyEstimator(gamma=gamma)

    def estimate(self,
                trajectories: List[Trajectory],
                eval_policy,
                behavior_policy,
                world_model,
                J: List[int] = [-1, 0, 1, 3, 5],
                kappa: int = 200,
                delta: float = 0.1) -> Dict[str, float]:
        """
        Run MAGIC algorithm

        Args:
            trajectories: List of trajectories
            eval_policy: Evaluation policy
            behavior_policy: Learned behavior policy
            world_model: Learned dynamics model
            J: Set of return lengths
            kappa: Number of bootstrap samples
            delta: Confidence level

        Returns:
            Dictionary with MAGIC estimate and diagnostics
        """
        print(f"\nRunning MAGIC with J={J}, kappa={kappa}, delta={delta}")

        n = len(trajectories)
        num_j = len(J)

        # Step 1: Compute all j-step returns
        print("Step 1: Computing j-step returns...")
        g_returns = {}  # g^(j) for each j
        per_traj_returns = {}  # Per-trajectory returns for each j

        for j in J:
            mean_return, traj_returns = self.estimator.compute_j_step_return(
                trajectories, eval_policy, behavior_policy, world_model, j
            )
            g_returns[j] = mean_return
            per_traj_returns[j] = np.array(traj_returns)
            print(f"  j={j}: {mean_return:.4f}")

        # Step 2: Estimate covariance matrix
        print("\nStep 2: Estimating covariance matrix...")
        Omega = np.zeros((num_j, num_j))

        for i, j_i in enumerate(J):
            for k, j_k in enumerate(J):
                # Sample covariance
                g_i = per_traj_returns[j_i]
                g_k = per_traj_returns[j_k]

                cov = np.cov(g_i, g_k)[0, 1]
                Omega[i, k] = cov

        print(f"Covariance matrix computed (condition number: {np.linalg.cond(Omega):.2e})")

        # Step 3: Bootstrap confidence interval for g^(inf) (WIS)
        print("\nStep 3: Computing bootstrap confidence interval...")

        # Use the longest horizon or WIS as g^(inf)
        g_inf_returns = per_traj_returns[J[-1]]  # Use last j as approximation

        # Bootstrap
        bootstrap_estimates = []
        for _ in range(kappa):
            # Resample trajectories
            indices = np.random.choice(n, size=n, replace=True)
            resampled_returns = g_inf_returns[indices]
            bootstrap_estimates.append(np.mean(resampled_returns))

        bootstrap_estimates = np.sort(bootstrap_estimates)

        # Percentile method
        lower_idx = int(0.05 * kappa)
        upper_idx = int(0.95 * kappa)

        WDR_estimate = np.mean(g_inf_returns)
        l = min(WDR_estimate, bootstrap_estimates[lower_idx])
        u = max(WDR_estimate, bootstrap_estimates[upper_idx])

        print(f"Bootstrap CI: [{l:.4f}, {u:.4f}]")

        # Step 4: Estimate bias vector
        print("\nStep 4: Estimating bias vector...")
        b_hat = np.zeros(num_j)

        for i, j in enumerate(J):
            g_j = g_returns[j]
            if g_j > u:
                b_hat[i] = g_j - u
            elif g_j < l:
                b_hat[i] = g_j - l
            else:
                b_hat[i] = 0.0

        print(f"Bias vector: {b_hat}")

        # Step 5: Solve quadratic program
        print("\nStep 5: Solving quadratic program...")

        # M = Omega + b * b^T
        M = Omega + np.outer(b_hat, b_hat)

        # Ensure M is positive semidefinite
        M = 0.5 * (M + M.T)  # Symmetrize
        eigenvalues = np.linalg.eigvals(M)
        if np.min(eigenvalues) < 0:
            print(f"Warning: M has negative eigenvalues, adding regularization")
            M = M + np.eye(num_j) * (abs(np.min(eigenvalues)) + 1e-6)

        # Solve: min x^T M x, s.t. x >= 0, sum(x) = 1
        x = cp.Variable(num_j)
        objective = cp.Minimize(cp.quad_form(x, M))
        constraints = [
            x >= 0,
            cp.sum(x) == 1
        ]
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.SCS, verbose=False)

            if x.value is None:
                print("Warning: QP solver failed, using uniform weights")
                x_star = np.ones(num_j) / num_j
            else:
                x_star = x.value
                print(f"Optimal weights: {x_star}")
        except Exception as e:
            print(f"Warning: QP solver error ({e}), using uniform weights")
            x_star = np.ones(num_j) / num_j

        # Step 6: Compute weighted combination
        print("\nStep 6: Computing final estimate...")
        magic_estimate = sum(x_star[i] * g_returns[J[i]] for i in range(num_j))

        print(f"\nMAGIC estimate: {magic_estimate:.4f}")

        return {
            "magic_estimate": magic_estimate,
            "j_estimates": g_returns,
            "weights": dict(zip(J, x_star)),
            "confidence_interval": (l, u),
            "covariance_matrix": Omega,
            "bias_vector": b_hat
        }


def MAGIC_Practical(buffer,
                   eval_policy,
                   world_model,
                   action_space,
                   J: List[int] = [-1, 0, 1, 3, 5],
                   kappa: int = 200,
                   delta: float = 0.1,
                   gamma: float = 0.99,
                   device: str = "cuda") -> Dict[str, float]:
    """
    Practical MAGIC implementation that takes raw offline data

    Args:
        buffer: ReplayBuffer with offline data
        eval_policy: Evaluation policy (policy to evaluate)
        world_model: Learned dynamics model (TransitionModel)
        action_space: Gym action space
        J: Set of return lengths
        kappa: Number of bootstrap samples
        delta: Confidence level
        gamma: Discount factor
        device: Device for computations

    Returns:
        Dictionary with MAGIC estimate and diagnostics
    """
    print("=" * 80)
    print("MAGIC Practical Algorithm")
    print("=" * 80)

    # Step 0a: Reconstruct trajectories
    print("\nStep 0a: Reconstructing trajectories from buffer...")
    reconstructor = TrajectoryReconstructor()
    trajectories = reconstructor.reconstruct_trajectories(buffer)

    # Step 0b: Learn behavior policy
    print("\nStep 0b: Learning behavior policy...")
    state_dim = buffer.observations.shape[1]
    action_dim = buffer.actions.shape[1]

    behavior_learner = BehaviorPolicyLearner(
        state_dim=state_dim,
        action_dim=action_dim,
        action_space=action_space,
        device=device
    )

    behavior_learner.learn(
        states=buffer.observations[:buffer.size],
        actions=buffer.actions[:buffer.size],
        epochs=50
    )

    # Step 0c: Use provided world model
    print("\nStep 0c: Using provided world model...")
    print(f"Model has {world_model.model.ensemble_size} ensemble models")

    # Run MAGIC
    print("\nRunning MAGIC algorithm...")
    magic = MAGICEstimator(gamma=gamma)
    results = magic.estimate(
        trajectories=trajectories,
        eval_policy=eval_policy,
        behavior_policy=behavior_learner,
        world_model=world_model,
        J=J,
        kappa=kappa,
        delta=delta
    )

    print("\n" + "=" * 80)
    print(f"Final MAGIC Estimate: {results['magic_estimate']:.4f}")
    print(f"95% CI: [{results['confidence_interval'][0]:.4f}, {results['confidence_interval'][1]:.4f}]")
    print("=" * 80)

    return results
