from utils import  rend_util
from src.utils import linear_rgb_to_srgb

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.interpolate import griddata

import torch
import torch.nn as nn
import torch.nn.functional as F


def diff_net_to_lin(x):
    # Convert diffuse component from network output to linear space
    # NOTE: Requires network output to be positive
    return x
    # return srgb_to_linear_rgb(x)

def spec_net_to_lin(x):
    # Convert specular component from network output to linear space
    # Opposite conversion is log(1+x)
    # It's a monotonic function so should be fine for negative 

    return x
    # return torch.exp(x)-1.
    # return linear_rgb_to_srgb_ub(x)
    # return spec_srgb_lin(x)

def apply_gamma_stokes(x):
    return linear_rgb_to_srgb(x)

def sdf_to_sigma(sdf: torch.Tensor, alpha, beta):
    # sdf *= -1 # NOTE: this will cause inplace opt.
    # sdf = -sdf
    # mask = sdf <= 0
    # cond1 = 0.5 * torch.exp(sdf / beta * mask.float())  # NOTE: torch.where will introduce 0*inf = nan
    # cond2 = 1 - 0.5 * torch.exp(-sdf / beta * (1-mask.float()))
    # # psi = torch.where(sdf <= 0, 0.5 * expsbeta, 1 - 0.5 / expsbeta)   # NOTE: exploding gradient
    # psi = torch.where(mask, cond1, cond2)
    
    """
    @ Section 3.1 in the paper. From sdf:d_{\Omega} to nerf's density:\sigma.
    work with arbitrary shape prefixes.
        sdf:    [...]
        
    """
    # -sdf when sdf > 0, sdf when sdf < 0
    exp = 0.5 * torch.exp(-torch.abs(sdf) / beta)
    psi = torch.where(sdf >= 0, exp, 1 - exp)

    return alpha * psi


def error_bound(d_vals, sdf, alpha, beta):
    """
    @ Section 3.3 in the paper. The error bound of a specific sampling.
    work with arbitrary shape prefixes.
    [..., N_pts] forms [..., N_pts-1] intervals, hence producing [..., N_pts-1] error bounds.
    Args:
        d_vals: [..., N_pts]
        sdf:    [..., N_pts]
    Return:
        bounds: [..., N_pts-1]
    """
    device = sdf.device
    sigma = sdf_to_sigma(sdf, alpha, beta)
    # [..., N_pts]
    sdf_abs_i = torch.abs(sdf)
    # [..., N_pts-1]
    # delta_i = (d_vals[..., 1:] - d_vals[..., :-1]) * rays_d.norm(dim=-1)[..., None]
    delta_i = d_vals[..., 1:] - d_vals[..., :-1]    # NOTE: already real depth
    # [..., N_pts-1]. R(t_k) of the starting point of the interval.
    R_t = torch.cat(
        [
            torch.zeros([*sdf.shape[:-1], 1], device=device), 
            torch.cumsum(sigma[..., :-1] * delta_i, dim=-1)
        ], dim=-1)[..., :-1]
    # [..., N_pts-1]
    d_i_star = torch.clamp_min(0.5 * (sdf_abs_i[..., :-1] + sdf_abs_i[..., 1:] - delta_i), 0.)
    # [..., N_pts-1]
    errors = alpha/(4*beta) * (delta_i**2) * torch.exp(-d_i_star / beta)
    # [..., N_pts-1]. E(t_{k+1}) of the ending point of the interval.
    errors_t = torch.cumsum(errors, dim=-1)
    # [..., N_pts-1]
    bounds = torch.exp(-R_t) * (torch.exp(errors_t) - 1.)
    # TODO: better solution
#     # NOTE: nan comes from 0 * inf
#     # NOTE: every situation where nan appears will also appears c * inf = "true" inf, so below solution is acceptable
    bounds[torch.isnan(bounds)] = np.inf
    return bounds


def fine_sample(implicit_surface_fn, init_dvals, rays_o, rays_d, 
                alpha_net, beta_net, far, 
                eps=0.05, max_iter:int=5, max_bisection:int=10, final_N_importance:int=64, N_up:int=128,
                perturb=True):
    """
    @ Section 3.4 in the paper.
    Args:
        implicit_surface_fn. sdf query function.
        init_dvals: [..., N_rays, N]
        rays_o:     [..., N_rays, 3]
        rays_d:     [..., N_rays, 3]
    Return:
        final_fine_dvals:   [..., N_rays, final_N_importance]
        beta:               [..., N_rays]. beta heat map
    """
    # NOTE: this algorithm is parallelized for every ray!!!
    with torch.no_grad():
        device = init_dvals.device
        prefix = init_dvals.shape[:-1]
        d_vals = init_dvals
        
        def query_sdf(d_vals_, rays_o_, rays_d_):
            pts = rays_o_[..., None, :] + rays_d_[..., None, :] * d_vals_[..., :, None]
            return implicit_surface_fn(pts)

        def opacity_invert_cdf_sample(d_vals_, sdf_, alpha_, beta_, N_importance=final_N_importance, det=not perturb):
            #-------------- final: d_vals, sdf, beta_net, alpha_net
            sigma = sdf_to_sigma(sdf_, alpha_, beta_)
            # bounds = error_bound(d_vals_, sdf_, alpha_net, beta_net)
            # delta_i = (d_vals_[..., 1:] - d_vals_[..., :-1]) * rays_d_.norm(dim=-1)[..., None]
            delta_i = d_vals_[..., 1:] - d_vals_[..., :-1]  # NOTE: already real depth
            R_t = torch.cat(
                [
                    torch.zeros([*sdf_.shape[:-1], 1], device=device), 
                    torch.cumsum(sigma[..., :-1] * delta_i, dim=-1)
                ], dim=-1)[..., :-1]
            #-------------- a fresh set of \hat{O}
            opacity_approx = 1 - torch.exp(-R_t)
            fine_dvals = rend_util.sample_cdf(d_vals_, opacity_approx, N_importance, det=det)
            return fine_dvals

        # final output storage.
        # being updated during the iterations of the algorithm
        final_fine_dvals = torch.zeros([*prefix, final_N_importance]).to(device)
        final_iter_usage = torch.zeros([*prefix]).to(device)

        #---------------- 
        # init beta+
        #---------------- 
        # [*prefix, 1]
        if not isinstance(far, torch.Tensor):
            far = far * torch.ones([*prefix, 1], device=device)
        beta = torch.sqrt((far**2) / (4 * (init_dvals.shape[-1]-1) * np.log(1+eps)))
        alpha = 1./beta
        # alpha = alpha_net
        # [*prefix, N]

        #---------------- 
        # first check of bound using network's current beta: B_{\mathcal{\tau}, \beta}
        #---------------- 
        # [*prefix]
        sdf = query_sdf(d_vals, rays_o, rays_d)
        net_bounds_max = error_bound(d_vals, sdf, alpha_net, beta_net).max(dim=-1).values
        mask = net_bounds_max > eps
        
        #---------------- 
        # first bound using beta+ : B_{\mathcal{\tau}, \beta_+}
        # [*prefix, N-1]
        bounds = error_bound(d_vals, sdf, alpha, beta)
        bounds_masked = bounds[mask]
        # NOTE: true for ANY ray that satisfy eps condition in the whole process
        final_converge_flag = torch.zeros([*prefix], device=device, dtype=torch.bool)
        
        # NOTE: these are the final fine sampling points for those rays that satisfy eps condition at the very beginning.
        if (~mask).sum() > 0:
            final_fine_dvals[~mask] = opacity_invert_cdf_sample(d_vals[~mask], sdf[~mask], alpha_net, beta_net)
            final_iter_usage[~mask] = 0
        final_converge_flag[~mask] = True
        
        cur_N = init_dvals.shape[-1]
        it_algo = 0
        #---------------- 
        # start algorithm
        #---------------- 
        while it_algo < max_iter:
            it_algo += 1
            #-----------------
            # the rays that not yet converged
            if mask.sum() > 0:
                #---------------- 
                # upsample the samples: \mathcal{\tau} <- upsample
                #---------------- 
                # [Masked, N_up]
                # NOTE: det = True should be more robust, forcing sampling points to be proportional with error bounds.
                # upsampled_d_vals_masked = rend_util.sample_pdf(d_vals[mask], bounds_masked, N_up, det=True)
                # NOTE: when using det=True, the head and the tail d_vals will always be appended, hence removed using [..., 1:-1]
                upsampled_d_vals_masked = rend_util.sample_pdf(d_vals[mask], bounds_masked, N_up+2, det=True)[..., 1:-1]
                
                # NOTE: for debugging
                # import matplotlib.pyplot as plt
                # ind = 0   # NOTE: this might not be the same ray as the 0-th rays may already converge before it reaches max_iter
                # fig, (ax1, ax2) = plt.subplots(2,1, figsize=(15, 15))
                # ax1.plot(d_vals[mask][ind].cpu(), sdf[mask][ind].cpu(), label='sdf')
                # ax1.legend()
                # ax2.step(d_vals[mask][ind].cpu()[..., :-1], bounds_masked[ind].cpu(), label='error bounds')
                # # ax2.step(d_vals[0].cpu()[..., :-1], error, label='error')
                # ax2.scatter(upsampled_d_vals_masked[ind].cpu(), y=np.zeros([N_up]), label='up sample')
                # ax2.legend()
                # ax2.set_title("it={}, beta_net={}, beta={:.3f}".format(it_algo, beta_net, beta[mask][ind].item()))
                # plt.show()
                
                d_vals = torch.cat([d_vals, torch.zeros([*prefix, N_up]).to(device)], dim=-1)
                sdf = torch.cat([sdf, torch.zeros([*prefix, N_up]).to(device)], dim=-1)
                # NOTE. concat and sort. work with any kind of dims of mask.
                d_vals_masked = d_vals[mask]
                sdf_masked = sdf[mask]
                d_vals_masked[..., cur_N:cur_N+N_up] = upsampled_d_vals_masked
                d_vals_masked, sort_indices_masked = torch.sort(d_vals_masked, dim=-1)
                sdf_masked[..., cur_N:cur_N+N_up] = query_sdf(upsampled_d_vals_masked, rays_o[mask], rays_d[mask])
                sdf_masked = torch.gather(sdf_masked, dim=-1, index=sort_indices_masked)
                d_vals[mask] = d_vals_masked
                sdf[mask] = sdf_masked
                # NOTE: another version of the above. only work with 1-dim mask.
                # d_vals[mask, cur_N:cur_N+N_up] = upsampled_d_vals_masked
                # d_vals[mask, :cur_N+N_up], sort_indices_masked = torch.sort(d_vals[mask, :cur_N+N_up], dim=-1)
                # sdf[mask, cur_N:cur_N+N_up] = query_sdf(upsampled_d_vals_masked, rays_o[mask], rays_d[mask])
                # sdf[mask, :cur_N+N_up] = torch.gather(sdf[mask, :cur_N+N_up], dim=-1, index=sort_indices_masked)
                cur_N += N_up

                #---------------- 
                # after upsample, check the bound using network's current beta: B_{\mathcal{\tau}, \beta}
                #---------------- 
                # NOTE: for the same iteration, the number of points of input rays are the same, (= cur_N), so they can be handled parallelized. 
                net_bounds_max[mask] = error_bound(d_vals[mask], sdf[mask], alpha_net, beta_net).max(dim=-1).values
                # NOTE: mask for those rays that still remains > eps after upsampling. 
                sub_mask_of_mask = net_bounds_max[mask] > eps
                # mask-the-mask approach. below 3 lines: final_converge_flag[mask][~sub_mask_of_mask] = True (this won't work in python)
                converged_mask = mask.clone()
                converged_mask[mask] = ~sub_mask_of_mask
                
                # NOTE: these are the final fine sampling points for those rays that >eps originally but <eps after upsampling.
                if converged_mask.sum() > 0:
                    final_converge_flag[converged_mask] = True
                    final_fine_dvals[converged_mask] = opacity_invert_cdf_sample(d_vals[converged_mask], sdf[converged_mask], alpha_net, beta_net)
                    final_iter_usage[converged_mask] = it_algo
                #---------------- 
                # using bisection method to find the new beta+ s.t. B_{\mathcal{\tau}, \beta+}==eps
                #---------------- 
                if (sub_mask_of_mask).sum() > 0:
                    # mask-the-mask approach
                    new_mask = mask.clone()
                    new_mask[mask] = sub_mask_of_mask
                    # [Submasked, 1]
                    beta_right = beta[new_mask]
                    beta_left = beta_net * torch.ones_like(beta_right, device=device)
                    d_vals_tmp = d_vals[new_mask]
                    sdf_tmp = sdf[new_mask]
                    #---------------- 
                    # Bisection iterations
                    for _ in range(max_bisection):
                        beta_tmp = 0.5 * (beta_left + beta_right)
                        alpha_tmp = 1./beta_tmp
                        # alpha_tmp = alpha_net
                        # [Submasked]
                        bounds_tmp_max = error_bound(d_vals_tmp, sdf_tmp, alpha_tmp, beta_tmp).max(dim=-1).values
                        beta_right[bounds_tmp_max <= eps] = beta_tmp[bounds_tmp_max <= eps]
                        beta_left[bounds_tmp_max > eps] = beta_tmp[bounds_tmp_max > eps]
                    beta[new_mask] = beta_right
                    alpha[new_mask] = 1./beta[new_mask]
                    
                    #---------------- 
                    # after upsample, the remained rays that not yet converged.
                    #---------------- 
                    bounds_masked = error_bound(d_vals_tmp, sdf_tmp, alpha[new_mask], beta[new_mask])
                    # bounds_masked = error_bound(d_vals_tmp, rays_d_tmp, sdf_tmp, alpha_net, beta[new_mask])
                    bounds_masked = torch.clamp(bounds_masked, 0, 1e5)  # NOTE: prevent INF caused NANs
                    
                    # mask = net_bounds_max > eps   # NOTE: the same as the following
                    mask = new_mask
                else:
                    break
            else:
                break
        
        #---------------- 
        # for rays that still not yet converged after max_iter, use the last beta+
        #---------------- 
        if (~final_converge_flag).sum() > 0:
            beta_plus = beta[~final_converge_flag]
            alpha_plus = 1./beta_plus
            # alpha_plus = alpha_net
            # NOTE: these are the final fine sampling points for those rays that still remains >eps in the end. 
            final_fine_dvals[~final_converge_flag] = opacity_invert_cdf_sample(d_vals[~final_converge_flag], sdf[~final_converge_flag], alpha_plus, beta_plus)
            final_iter_usage[~final_converge_flag] = -1
        beta[final_converge_flag] = beta_net
        return final_fine_dvals, beta, final_iter_usage