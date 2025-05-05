
from pydrake.all import PiecewisePolynomial
import numpy as np
import torch
import math
from torch.distributions import MultivariateNormal
import time
# example usage see streaming_flow_policy/test_vt.ipynb

def get_traj_seq(arr, T):
    """
    Given an array of shape (n, 5), return an array of shape (n, T+3, 5) where we take the consecutive sequence of length T+3 of the array and pad the sequence in the beginning and at the end with values same as the first entry and last entry.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array of shape (n, 5)
    T : int
        Length of the sequence

    Returns
    -------
    numpy.ndarray
        Output array of shape (n, T+3, 5)
    """
    n = arr.shape[0]
    sequences = np.array([np.pad(arr[max(0, i-1):min(n, i+T+2)], ((max(0, 1-i), max(0, i+T+2-n)), (0, 0)), mode='edge') for i in range(n)])
    return sequences
# # Example usage
# arr = np.array([[1, 2, 3, 4, 5],
#                 [6, 7, 8, 9, 10],
#                 [11, 12, 13, 14, 15],
#                 [16, 17, 18, 19, 20],
#                 [21, 22, 23, 24, 25]])
# T = 3

# result = get_traj_seq(arr, T)
# print(result)

def tensor_to_trajectory(tensor, T):
    """
    Convert a torch tensor of shape (T+1, 5) to a pydrake Trajectory object using PiecewisePolynomial.FirstOrderHold().
    
    Parameters
    ----------
    tensor : np,ndarray
        Input array of shape (T+1, 5)
    
    Returns
    -------
    traj : pydrake.trajectories.PiecewisePolynomial
        Trajectory object
    """
    if isinstance(tensor, torch.Tensor): 
        data = tensor.detach().cpu().numpy()
    else:
        data = tensor
    # Create the corresponding time array
    if T != data.shape[0] - 1:
        raise ValueError(f"the prediction horizon T should be equal to tensor.shape[0] - 1, but now have T= {T} and data.shape[0] = {data.shape[0]}")
    
    time = np.linspace(0, 1, T + 1)
    
    # Create the PiecewisePolynomial trajectory
    traj = PiecewisePolynomial.FirstOrderHold(time, data.T)
    
    return traj

def get_latent_training_data(traj_array, T, t, sigma0 = 0.1, sigma1 = 0.1, k=0):

    """
    Convert a torch tensor to a pydrake Trajectory object and get the state x_t and derivative v_t at time t.

    Parameters
    ----------
    traj_array : np.ndarray
        Input array of shape (T+2, 5)
    T : int
        Length of the sequence
    t : float
        Time at which to get the state and derivative

    Returns
    -------
    x_t : numpy.ndarray
        State x_t at time t
    v_t : numpy.ndarray
        Derivative v_t at time t
    """
    action_dim = traj_array.shape[-1] #traj_array:(seq_len, action_dim)
    latent_helper = StreamingFlowPolicyLatent(dim = action_dim, trajectories = [],prior=[],
                                            σ0 = sigma0, σ1 = sigma1, k=k)
    # Convert tensor to trajectory
    trajectory = tensor_to_trajectory(traj_array, T)
    μt, Σt = latent_helper.μΣt(trajectory, t.cpu())  # (*BS, X) and (*BS, X, X)
    Σt = Σt + 1e-6 * torch.eye(Σt.shape[-1], device=Σt.device)  # (*BS, X, X)
    dist = MultivariateNormal(loc=μt, covariance_matrix=Σt)  # BS=(*BS) ES=(X,)
    x_t = dist.sample()  # (*BS, 2D)
    # print('x_t', x_t.shape)
    v_t = latent_helper.v_conditional(trajectory, x_t, t.cpu()) # (*BS, 2D)
    # print('vt', v_t.shape)
    return x_t, v_t

def get_x_v_o(traj_array, T, t):
    """
    Convert a torch tensor to a pydrake Trajectory object and get the state x_t and derivative v_t at time t.

    Parameters
    ----------
    traj_array : np.ndarray
        Input array of shape (T+2, 5)
    T : int
        Length of the sequence
    t : float
        Time at which to get the state and derivative

    Returns
    -------
    x_t : numpy.ndarray
        State x_t at time t
    v_t : numpy.ndarray
        Derivative v_t at time t
    """
    # Convert tensor to trajectory
    trajectory = tensor_to_trajectory(traj_array, T)
    
    # Get the state x_t from the trajectory at time t
    x_t = trajectory.value(t) #get_x_t(trajectory, t)
    
    # Get the derivative v_t from the trajectory at time t
    v_t = trajectory.EvalDerivative(t)  #get_v_t(trajectory, t)

    return x_t, v_t

# # def get_x_v_fast(x_seq, T, t, device):
#     # x_seq: B, seq_len, action_dim -> seq_len, B, action_dim
#     x_seq_np = x_seq.permute(1, 0, 2).detach().cpu().numpy()
#     time = np.linspace(0, 1, T + 1)
#     trajectory = PiecewisePolynomial.FirstOrderHold(time, x_seq_np)
#     x_t = trajectory.value(t) #get_x_t(trajectory, t)
#     v_t = trajectory.EvalDerivative(t)  #get_v_t(trajectory, t)
#     return torch.from_numpy(x_t).to(device), torch.from_numpy(v_t).to(device)

def biased_sample(x_seq, t, T, prob=0.9):
    B, seq_len, _ = x_seq.shape
    device = x_seq.device
    dtype  = t.dtype

    gr = x_seq[..., -1]               # [B, seq_len]
    # print('gr', gr)

    # has_pos = (gr ==  1).any(dim=1)   # [B]
    # has_neg = (gr == -1).any(dim=1)   # [B]
    has_pos = (gr > 0).any(dim=1)   # [B]
    has_neg = (gr < 0).any(dim=1)   # [B]
    rnd_ok  = torch.rand(B, device=device) < 0.9

    # Only resample those that truly have both signs and pass the random check
    valid = has_pos & has_neg & rnd_ok  # [B]

    if valid.any():
        # print('gripper transition')
        # Compute the switch mask once
        switch_mask = gr[:, :-1] != gr[:, 1:]           # [B, seq_len-1]
        # First switch index (we know there is at least one for `valid` rows)
        first_sw = switch_mask.float().argmax(dim=1)    # [B]

        invT    = 1.0 / T
        t_start = first_sw.to(dtype) * invT             # [B]
        t_end   = (first_sw.to(dtype) + 1) * invT       # [B]

        u = torch.rand(B, device=device, dtype=dtype)   # [B]

        t[valid] = u[valid] * (t_end[valid] - t_start[valid]) \
                       + t_start[valid]
        # new_t = t.clone()
        # new_t[valid] = u[valid] * (t_end[valid] - t_start[valid]) \
        #                + t_start[valid]
        # t = new_t

    return t


def fast_get_x_v_tensor(x_seq, t, T, gripper_normalize=1.0, gripper_velocity=False):
    """
    Vectorized computation of positions and velocities for a batch of trajectories.
    x_seq: Tensor of shape (B, T+1, D)
    T: horion; should be 16 all the time
    t: Tensor of shape (B,) with values in [0, 1]
    gripper_normalize: scalar factor to divide the last dimension of v_t (default 1.0)
    Returns:
        x_t: Tensor of shape (B, D) - interpolated positions at times t
        v_t: Tensor of shape (B, D) - velocities at times t
    """
    B, seq_len, D = x_seq.shape
    if T !=  seq_len - 1:
        raise ValueError(f"the prediction horizon T should be equal to tensor.shape[0] - 1, but now have T= {T} and data.shape[0] = {seq_len}")

    # compute interpolation indices and weights
    scaled_t = t * T
    k = torch.clamp(scaled_t.floor().long(), 0, T - 1)
    alpha = scaled_t - k.float()

    # gather the two timesteps
    batch_idx = torch.arange(B, device=x_seq.device)
    seq_k  = x_seq[batch_idx, k, :]
    seq_k1 = x_seq[batch_idx, k + 1, :]

    # interpolate position
    x_t = seq_k + alpha.unsqueeze(-1) * (seq_k1 - seq_k)
    # compute velocity
    v_t = (seq_k1 - seq_k) * T
    # print('seq_k1 - seq_k', (seq_k1 - seq_k)[0])
    # print('ut_nonoise', v_t[0])

    # apply gripper normalization on the last dimension if requested
    if gripper_velocity:
        # Set v_t[..., -1] to +1 if > 0, -1 if < 0, 0 if == 0
        v_last = v_t[..., -1]
        v_t[..., -1] = torch.where(v_last > 0, torch.tensor(1.28, device=v_t.device),
                            torch.where(v_last < 0, torch.tensor(-1.28, device=v_t.device),
                                         torch.tensor(0.0, device=v_t.device)))
        # print('v_t', v_t[0, -1])
        # TODO: need to check if this is correct
        # if t is in segment with gripper actions (a, b), 
        # if b is +1 the gripper velocity for t should be treated as +32 
        # find the segment index for each sample (ceiling of scaled_t)
        # time_idx = torch.clamp(scaled_t.ceil().long(), 0, T)
        # gripper_action = x_seq[batch_idx, time_idx, -1]  # shape (B,)
        # pos_mask = gripper_action > 0 # shape (B,)

        # # set +32 when action >0, else -32
        # v_t[pos_mask,     -1] =  32
        # v_t[~pos_mask,    -1] = -32
    if gripper_normalize != 1.0:
        v_t[..., -1] = v_t[..., -1] / gripper_normalize
    # print('ut no noise', v_t[0, -1])
    return x_t, v_t

def get_total_xt_ut_ot(x_seq, t, T = 16, k = 0, sigma = 0.01, device = torch.device("cuda"), 
                       biased_gripper = False, gripper_velocity = False, gripper_no_noise = False, gripper_normalize = 1,
                       biased_prob = 0.9, 
                       latent = False, sigma0 = 0.1, sigma1 = 0.1):
    '''
    x_seq: (B, seq_len, action_dim) action traj
    t: (B,) tensor of random time steps in Uniform(0, 1)
    biased: if True, if the training chunk contains a switch between k and k+1, 
    instead of sampling t uniformly between 0 and 1, 50% of the time sample t between k and k+1
    sample xt = q(t) ~ N(q̃(t), σ₀ exp(-kt))
    calculate velocity at time t: ut = -k * (qt - q̃t) + ṽt where (qt - q̃t) = added_noise
    '''
    # x_seq: B, seq_len, action_dim
    # t: (B,)
    x_t_list, v_t_list = [], []
    # x_t, v_t = get_x_v_fast(x_seq, T, t, device)
    if biased_gripper: t = biased_sample(x_seq, t, T, prob=biased_prob)
    xt, ut = fast_get_x_v_tensor(x_seq, t, T, gripper_normalize=gripper_normalize, gripper_velocity=gripper_velocity)  # (B, D)
    # print('ut', ut[0,:])
    # if not latent:
    added_noise = sigma * torch.exp(-k*t).unsqueeze(1) * torch.randn_like(xt)
    xt = xt + added_noise

    if gripper_no_noise:
        added_noise[:, -1] = 0
    # print('v change', k*added_noise[0, 0])
    ut = -k * added_noise + ut
    # print('ut noise', ut[0,:])
    # print('\n')

    return xt, ut #(batch_size, action_dim)
