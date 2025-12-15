import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import torchvision.transforms as transforms


#############################################
# 1. 你的 GT preprocessing pipeline
#############################################

def build_preprocess(image_size):
    return transforms.Compose([
        transforms.ToTensor(),                   # [0,1]
        transforms.Resize(image_size),           # 等比例缩放（保持长宽比）
        transforms.CenterCrop(image_size),       # 居中裁剪成正方形
        transforms.Normalize((0.5, 0.5, 0.5), 
                             (0.5, 0.5, 0.5))    # [0,1] → [-1,1]
    ])


#############################################
# 2. 加载 GT 图像（保持长宽比）
#############################################

def load_gt(path, transform):
    img = Image.open(path).convert("RGB")
    return transform(img)   # CHW, [-1,1]


#############################################
# 3. 加载 Pred（假设已经是 [-1,1] 或 [0,1]）
#############################################

def load_pred(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2,0,1)) * 2 - 1   # 转成 [-1,1]


#############################################
# 4. Metric Functions
#############################################

def compute_psnr(gt, pred):
    # [-1,1] → [0,1]
    gt = torch.clamp((gt + 1) / 2, 0, 1).permute(1,2,0).cpu().numpy()
    pred = torch.clamp((pred + 1) / 2, 0, 1).permute(1,2,0).cpu().numpy()
    return peak_signal_noise_ratio(gt, pred, data_range=1.0)


def compute_ssim(gt, pred):
    gt = torch.clamp((gt + 1) / 2, 0, 1).permute(1,2,0).cpu().numpy()
    pred = torch.clamp((pred + 1) / 2, 0, 1).permute(1,2,0).cpu().numpy()
    return structural_similarity(gt, pred, data_range=1.0, channel_axis=2)


#############################################
# 5. 主评估函数
#############################################

def evaluate_folder(gt_folder, pred_folder, image_size=256):

    gt_folder = Path(gt_folder)
    pred_folder = Path(pred_folder)

    transform_gt = build_preprocess(image_size)

    gt_files = sorted(gt_folder.glob("*.png"))
    pred_files = sorted(pred_folder.glob("*.png"))

    assert len(gt_files) == len(pred_files), "数量不匹配！"

    lpips_fn = lpips.LPIPS(net='vgg').cuda()

    psnr_all, ssim_all, lpips_all = [], [], []

    for gt_path, pred_path in zip(gt_files, pred_files):
        gt = load_gt(gt_path, transform_gt)
        pred = load_pred(pred_path)

        # Shape 必须一致（C,H,W）
        assert gt.shape == pred.shape, f"Shape mismatch: {gt_path.name}"

        psnr_all.append(compute_psnr(gt, pred))
        ssim_all.append(compute_ssim(gt, pred))

        gt_t = gt.unsqueeze(0).cuda()
        pred_t = pred.unsqueeze(0).cuda()
        lp = lpips_fn(gt_t, pred_t).item()
        lpips_all.append(lp)

    print("==== Final Metrics ====")
    print(f"PSNR : {np.mean(psnr_all):.4f}")
    print(f"SSIM : {np.mean(ssim_all):.4f}")
    print(f"LPIPS: {np.mean(lpips_all):.4f}")

    return psnr_all, ssim_all, lpips_all


#############################################
# 6. CLI
#############################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--image_size", default=256, type=int)
    args = parser.parse_args()

    evaluate_folder(args.gt, args.pred, args.image_size)
