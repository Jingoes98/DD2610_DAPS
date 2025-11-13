import json
import yaml
import torch
from torchvision.utils import save_image
from forward_operator import get_operator
from data import get_dataset
from sampler import get_sampler, Trajectory
from model import get_model
from eval import get_eval_fn, get_eval_fn_cmp, Evaluator
from torch.nn.functional import interpolate
from pathlib import Path
from omegaconf import OmegaConf
from evaluate_fid import calculate_fid
from torch.utils.data import DataLoader
import hydra
import wandb
import setproctitle
from PIL import Image
import numpy as np
import imageio

def resize(y, x, task_name):
    """
        Visualization Only: resize measurement y according to original signal image x
    """
    if y.shape != x.shape:
        ry = interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
    else:
        ry = y
    if task_name == 'phase_retrieval':
        def norm_01(y):
            tmp = (y - y.mean()) / y.std() # normalize to mean 0, std 1
            tmp = tmp.clip(-0.5, 0.5) * 3  # clip and amplify
            return tmp

        ry = norm_01(ry) * 2 - 1 # normalize to [-1, 1] range
    return ry

def safe_dir(dir):
    """
        get (or create) a directory
    """
    if not Path(dir).exists():
        Path(dir).mkdir()
    return Path(dir)

def norm(x):
    """
        normalize data to [0, 1] range
    """
    return (x * 0.5 + 0.5).clip(0, 1)

def tensor_to_pils(x):
    """
        [B, C, H, W] tensor -> list of pil images 
    """
    pils = []
    for x_ in x:
        np_x = norm(x_).permute(1, 2, 0).cpu().numpy() * 255 # [H, W, C] for PIL
        np_x = np_x.astype(np.uint8) # to uint8
        pil_x = Image.fromarray(np_x) # to pil image
        pils.append(pil_x)
    return pils

def tensor_to_numpy(x):
    """
        [B, C, H, W] tensor -> [B, H, W, C] numpy uint8 images
    """
    np_images = norm(x).permute(0, 2, 3, 1).cpu().numpy() * 255
    return np_images.astype(np.uint8)

def save_mp4_video(gt, y, x0hat_traj, x0y_traj, xt_traj, output_path, fps=24, sec=5, space=4):
    """
        stack and save trajectory as mp4 video
    """
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
    ix, iy = x0hat_traj.shape[-2:]
    reindex = np.linspace(0, len(xt_traj) - 1, sec * fps).astype(int)
    np_x0hat_traj = tensor_to_numpy(x0hat_traj[reindex])
    np_x0y_traj = tensor_to_numpy(x0y_traj[reindex])
    np_xt_traj = tensor_to_numpy(xt_traj[reindex])
    np_y = tensor_to_numpy(y[None])[0]
    np_gt = tensor_to_numpy(gt[None])[0]
    for x0hat, x0y, xt in zip(np_x0hat_traj, np_x0y_traj, np_xt_traj):
        canvas = np.ones((ix, 5 * iy + 4 * space, 3), dtype=np.uint8) * 255
        cx = cy = 0
        canvas[cx:cx + ix, cy:cy + iy] = np_y

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = np_gt

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = x0y

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = x0hat

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = xt
        writer.append_data(canvas)
    writer.close()

def sample_in_batch(sampler, model, x_start, operator, y, evaluator, verbose, record, batch_size, gt, args, root, run_id):
    """
        posterior sampling in batch
        - x_start: initial noise samples [N, C, H, W]
        - operator: forward operator for conditioning
        - y: measurements [N, C, H, W]
        - evaluator: compute evaluation metrics
        - batch_size: batch size for sampling
    """
    samples = []
    trajs = []
    for s in range(0, len(x_start), batch_size):
        # update evaluator to correct batch index
        cur_x_start = x_start[s:s + batch_size]
        cur_y = y[s:s + batch_size]
        cur_gt = gt[s: s + batch_size]
        cur_samples = sampler.sample(model, cur_x_start, operator, cur_y, evaluator, verbose=verbose, record=record, gt=cur_gt)

        samples.append(cur_samples)
        if record:
            cur_trajs = sampler.trajectory.compile()
            trajs.append(cur_trajs)

        # log individual sample instances
        if args.save_samples:
            pil_image_list = tensor_to_pils(cur_samples)
            image_dir = safe_dir(root / 'samples')
            for idx in range(batch_size):
                image_path = image_dir / '{:05d}_run{:04d}.png'.format(idx+s, run_id)
                pil_image_list[idx].save(str(image_path))

        # log sampling trajectory and mp4 video
        if args.save_traj:
            traj_dir = safe_dir(root / 'trajectory')
            # save mp4 video for trajectories
            x0hat_traj = cur_trajs.tensor_data['x0hat']
            x0y_traj = cur_trajs.tensor_data['x0y']
            xt_traj = cur_trajs.tensor_data['xt']
            cur_resized_y = resize(cur_y, cur_samples, args.task[args.task_group].operator.name)
            slices = np.linspace(0, len(x0hat_traj)-1, 10).astype(int)
            slices = np.unique(slices)
            for idx in range(batch_size):
                if args.save_traj_video:
                    video_path = str(traj_dir / '{:05d}_run{:04d}.mp4'.format(idx+s, run_id))
                    save_mp4_video(cur_samples[idx], cur_resized_y[idx], x0hat_traj[:, idx], x0y_traj[:, idx], xt_traj[:, idx], video_path)
                # save long grid images
                selected_traj_grid = torch.cat([x0y_traj[slices, idx], x0hat_traj[slices, idx], xt_traj[slices, idx]], dim=0)
                traj_grid_path = str(traj_dir / '{:05d}_run{:04d}.png'.format(idx+s, run_id))
                save_image(selected_traj_grid * 0.5 + 0.5, fp=traj_grid_path, nrow=len(slices))
        
    if record:
        trajs = Trajectory.merge(trajs)
    return torch.cat(samples, dim=0), trajs
