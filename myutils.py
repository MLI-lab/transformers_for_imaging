import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
# from prettytable import PrettyTable

def divide_img(img, patch_size):
    """Divides image into tensor of image patches
    Args:
        img: batch of images, torch.tensor, e.g. [batch_size, channels, 32, 32]
        patch_size: patch size, tuple e.g. (4,4)
    
    Returns:
        A torch.tensor of stacked stacked flattened image patches, e.g. [batch_size, 64, channels, 4, 4]
    """
    height = img.shape[2] 
    width = img.shape[3]
    patch_size_h = patch_size[0]
    patch_size_w = patch_size[1]
    A = []
    for i in range(int(height/patch_size_h)):
        for j in range(int(width/patch_size_w)):
            A.append(img[:,:,i*patch_size_h:i*patch_size_h+patch_size_h,j*patch_size_w:j*patch_size_w+patch_size_w])
    return torch.stack(A).permute(1,0,2,3,4)

def compose_img(img_patches, img_size, patch_size):
    """Composes an image given a tensor of patches
    Args:
        img_patches: batches of image patch sequences, torch.tensor, e.g. [batch_size, 64, channels, 4, 4]
        img_size: image size, tuple  (channels, height, width)
        patch_size: patch size, tuple e.g. (4,4)
    
    Returns:
        A torch.tensor of batch of images, e.g. [batch_size, channels, 32, 32]
    """
    channels = img_size[0]
    height = img_size[1]
    width = img_size[2]    
    patch_size_h = patch_size[0]
    patch_size_w = patch_size[1]
    A = torch.zeros(img_patches.shape[0],channels,height,width).to(img_patches.device)
    for i in range(int(height/patch_size_h)):
        for j in range(int(width/patch_size_w)):
            A[:,:,i*patch_size_h:i*patch_size_h+patch_size_h,j*patch_size_w:j*patch_size_w+patch_size_w] = img_patches[:,i*int(width/patch_size_w)+j]
            
    return A

def imshow(img):
    """function to show an normalized image (-1,1) torch tensor"""
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def compute_psnr(img1, img2, maxval):
    """Computes PSNR in dB"""
    mse = torch.mean((img1 - img2) ** 2)

    return (10 * torch.log10(maxval / mse)).item()

def compute_psnr(img1, img2, maxval):
    """Computes PSNR in dB"""
    err = ((img1 - img2) ** 2).view(img1.shape[0],img1.shape[1]*img1.shape[2])
    mse = torch.mean(err, dim=1)
    return (10 * torch.log10(maxval**2 / mse))

class SSIM(nn.Module):
    """
    SSIM module. From fastMRI SSIM loss
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D
        S = S.view(S.shape[0],S.shape[-2]*S.shape[-1])
        
        return S.mean(dim=1)

class PSNRLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        # Y is target
        err = ((X - Y) ** 2).reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
        mse = torch.mean(err, dim=1)
        return 100-(10 * torch.log10(data_range**2 / mse)).mean()
    
class PSNR(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        # Y is target
        err = ((X - Y) ** 2).reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
        mse = torch.mean(err, dim=1)
        return (10 * torch.log10(data_range**2 / mse))

class NMSE(nn.Module):   
    def __init__(self, ):
        super().__init__()    
        
    def forward(self, X, Y):
        # Y is target
        err = (Y - X).reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
        err = (err**2).sum(dim=-1)
        den = (Y.reshape(Y.shape[0], Y.shape[1]*Y.shape[2]*Y.shape[3]) ** 2).sum(dim=-1)
        return err/den
    
class NMSELoss(NMSE):   
    def __init__(self, ):
        super().__init__()    
        
    def __call__(self, X, Y):

        return super().forward(X,Y).mean()

class L1Loss(nn.Module):
    def __init__(self, ):
        super().__init__()    
        
    def forward(self, X, Y, data_range):
        err = (Y - X).reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
        err = err.abs().sum(dim=-1)
        
        return (err/data_range).mean()
    
    
# def count_parameters(model):
#     """"Model parameters count"""
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad: continue
#         param = parameter.numel()
#         table.add_row([name, param])
#         total_params+=param
#     print(table)
#     print("Total Trainable Params: {total_params}")
#     return total_params