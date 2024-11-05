#@title Sampling (double click to expand or collapse)

from torchvision.utils import make_grid
import torch
import tqdm
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm
from PIL import Image, ImageDraw

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import *
from barrier_function import BarrierFunction

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_gamma(t, a=0.0001, alpha=0.01):
  return a * (1 - np.exp(-alpha * t))

#@title Define the Euler-Maruyama sampler (double click to expand or collapse)

## The number of sampling steps.
num_steps =  1000#@param {'type':'integer'}
def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           batch_size=64, 
                           num_steps=num_steps,
                           eps=1e-3,
                           V=None):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns:
    Samples.    
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
    * marginal_prob_std(t)[:, None, None, None]
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x.clone()
  x_cons = init_x.clone()
  nr_it = -1

  with torch.no_grad():
    for time_step in tqdm(time_steps):
      nr_it += 1      
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(batch_time_step)

      mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
      x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)  
      
      gamma = 0.001#get_gamma(nr_it)
      barrier_gradient = V.barrier_gradient(x_cons)
      mean_x_cons = x_cons + (g**2)[:, None, None, None] * (score_model(x_cons, batch_time_step) + gamma*barrier_gradient)* step_size
      x_cons = mean_x_cons + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x_cons)      
  
  # Do not include any noise in the last sampling step.
  return mean_x, mean_x_cons

#@title Define the Predictor-Corrector sampler (double click to expand or collapse)
signal_to_noise_ratio = 0.16 #@param {'type':'number'}
## The number of sampling steps.
num_steps =  500#@param {'type':'integer'}
def pc_sampler(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64, 
               num_steps=num_steps, 
               snr=signal_to_noise_ratio,
               eps=1e-3):
  """Generate samples from score-based models with Predictor-Corrector method.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient 
      of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.    
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns: 
    Samples.
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
  time_steps = np.linspace(1., eps, num_steps)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    for time_step in tqdm(time_steps):      
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      # Corrector step (Langevin MCMC)
      grad = score_model(x, batch_time_step)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = np.sqrt(np.prod(x.shape[1:]))
      langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
      x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

      # Predictor step (Euler-Maruyama)
      g = diffusion_coeff(batch_time_step)
      x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
      x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      
    
    # The last step does not include any noise
    return x_mean
  
#@title Define the ODE sampler (double click to expand or collapse)
## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5 #@param {'type': 'number'}
def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64, 
                atol=error_tolerance, 
                rtol=error_tolerance, 
                z=None,
                eps=1e-3):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  t = torch.ones(batch_size, device=device)
  # Create the latent code
  if z is None:
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
      * marginal_prob_std(t)[:, None, None, None]
  else:
    init_x = z
    
  shape = init_x.shape

  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
    with torch.no_grad():    
      score = score_model(sample, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)
  
  def ode_func(t, x):        
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t    
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)
  
  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
  print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

  return x

## Load the pre-trained checkpoint from disk.

sigma =  25.0#@param {'type':'number'}
score_model, marginal_prob_std_fn, diffusion_coeff_fn = get_components(sigma)

ckpt = torch.load('data/ckpt.pth', map_location=device)
score_model.load_state_dict(ckpt)

sample_batch_size = 64 #@param {'type':'integer'}
sampler = Euler_Maruyama_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

V = BarrierFunction()

## Generate samples using the specified sampler.
samples, samples_cons = sampler(score_model, 
                        marginal_prob_std_fn,
                        diffusion_coeff_fn, 
                        sample_batch_size,
                        V=V)

## Sample visualization.
samples = samples.clamp(0.0, 1.0)
samples_cons = samples_cons.clamp(0.0, 1.0)

processed_images = []

for sample in samples_cons:
    img = sample.squeeze(0).cpu().numpy() * 255  # 转换为 [0, 255] 的灰度值
    img = Image.fromarray(img.astype(np.uint8))
    img = img.convert("RGB")

    draw = ImageDraw.Draw(img)
    draw.ellipse(
        [
            (V.center[0] - V.radius, V.center[1] - V.radius),
            (V.center[0] + V.radius, V.center[1] + V.radius)
        ],
        outline="red", width=1
    )

    processed_images.append(torch.from_numpy(np.array(img)).permute(2, 0, 1) / 255.0)
processed_images_tensor = torch.stack(processed_images)

sample_grid1 = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
sample_grid2 = make_grid(processed_images_tensor, nrow=int(np.sqrt(samples.size(0))))


plt.figure(figsize=(12, 6)) 

plt.subplot(1, 2, 1)
plt.axis('off')
plt.imshow(sample_grid1.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.title("Original Samples")

plt.subplot(1, 2, 2)
plt.axis('off')
plt.imshow(sample_grid2.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.title("Samples with Barrier")

plt.show()