from curses import use_env
from models.base import (ImplicitSurface, NeRF, RadianceNet, 
                         DiffuseNet, IntEnvMapNet,
                         MaskNet, RoughNet)
from models.two_bounce_base import (LightFieldNetV1)
from models.mip_nerf_base import MipNerf 
from models.nerf import Embedding, render_rays

from utils import io_util, train_util, rend_util
from utils.logger import Logger
from src.polarization import stokes_from_normal_rad, get_fresnel, stokes_fac_from_normal
from src.utils import srgb_to_linear_rgb, linear_rgb_to_srgb, spec_srgb_lin, linear_rgb_to_srgb_ub
from src.spherical import get_ide

import math
import imageio
import copy
import functools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.interpolate import griddata
from dotdict import dotdict

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.two_bounce_utils import compute_radii, get_virtual_viewpoint_lens_approx, distloss
from utils.virtual_cam import get_virtual_cone_params_sphere
from .volsdf_utils import * 

class VolSDF(nn.Module):
    def __init__(self,
                 beta_init=0.1,
                 speed_factor=1.0,
                 input_ch=3,
                 W_geo_feat=-1,
                 obj_bounding_radius=3.0,
                 use_nerfplusplus=False,
                 polarized=False,
                 pol_render_type='stokes_accum',
                 only_diffuse=False,
                 use_env_mlp='no_envmap_MLP',
                 env_mlp_type='no_fres',
                 fine_spec=False,
                 sh_embed=False,
                 disable_fres=False,
                 surface_cfg=dict(),
                 radiance_cfg=dict(),
                 two_bounce_cfg=dict()):
        super().__init__()
        
        self.speed_factor = speed_factor
        ln_beta_init = np.log(beta_init) / self.speed_factor
        self.ln_beta = nn.Parameter(data=torch.Tensor([ln_beta_init]), requires_grad=True)
        # self.beta = nn.Parameter(data=torch.Tensor([beta_init]), requires_grad=True)

        self.mask_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.use_sphere_bg = not use_nerfplusplus
        self.obj_bounding_radius = obj_bounding_radius
        self.implicit_surface = ImplicitSurface(
            W_geo_feat=W_geo_feat, input_ch=input_ch, obj_bounding_size=obj_bounding_radius, **surface_cfg)

        self.use_env_mlp = use_env_mlp
        self.env_mlp_type = env_mlp_type
        self.fine_spec = fine_spec
        self.sh_embed = sh_embed
        self.disable_fres = disable_fres

        self.sampling_type = "5dreflected_only"

        if W_geo_feat < 0:
            W_geo_feat = self.implicit_surface.W

        ##########
        # We have a default mask and diffuse net
        if (self.sh_embed):
            sh_coeff_fn = get_ide
        else:
            sh_coeff_fn = None

        self.mask_net = MaskNet(
                            D=radiance_cfg["D"],
                            W=radiance_cfg["W"],
                            skips=radiance_cfg["skips"],
                            W_geo_feat=W_geo_feat, 
                            embed_multires=radiance_cfg["embed_multires"],
                            final_act='sigmoid')
        if not two_bounce_cfg['remove_diffuse_component']:
            self.diffuse_net = DiffuseNet(
                            D=radiance_cfg["D"],
                            W=radiance_cfg["W"],
                            skips=radiance_cfg["skips"],
                            W_geo_feat=W_geo_feat, 
                            embed_multires=radiance_cfg["embed_multires"],
                            final_act='sigmoid')
        else:
            print("Removing Diffuse Net!")
            self.diffuse_net = None
        ##########  
        if (self.use_env_mlp == "rough_mask_envmap_MLP"):
            # default case for PANDORA for baselines
            if not only_diffuse:
                self.rough_net = RoughNet(
                                    D=radiance_cfg["D"],
                                    W=radiance_cfg["W"],
                                    skips=radiance_cfg["skips"],
                                    W_geo_feat=W_geo_feat, 
                                    embed_multires=radiance_cfg["embed_multires"],
                                    final_act='softplus')
                if self.env_mlp_type == 'no_fres':
                    self.specular_net = IntEnvMapNet(
                                            embedding_fn=sh_coeff_fn,
                                            D=radiance_cfg["D"],
                                            W=radiance_cfg["W"],
                                            skips=radiance_cfg["skips"],
                                            embed_multires_view=radiance_cfg["embed_multires_view"],
                                            final_act='softplus')
                else:
                    raise ValueError("no_fres is only allowed to run the default PANDORA case.")
                    
        elif self.use_env_mlp == "simple_env_5dscene_repres":
            # (x, IDE) -> F -> spec_color; there is no intengration or rendering! just a function
            if not only_diffuse:
                self.rough_net = RoughNet(
                                    D=radiance_cfg["D"],
                                    W=radiance_cfg["W"],
                                    skips=radiance_cfg["skips"],
                                    W_geo_feat=W_geo_feat, 
                                    embed_multires=radiance_cfg["embed_multires"],
                                    final_act='softplus')
                self.specular_net = LightFieldNetV1(                                            
                                            embedding_fn=sh_coeff_fn,
                                            D=radiance_cfg["D"],
                                            W=radiance_cfg["W"],
                                            skips=radiance_cfg["skips"],
                                            use_positional_embedding=True,
                                            embed_multires_view=radiance_cfg["embed_multires_view"],
                                            final_act='softplus')

        elif self.use_env_mlp == "env_as_5d_radiance_field_nerf":
            # implements a 5d radiance field
            # with inputs (x, IDE) -> F -> (radiance, sigma); We integrate the ray to get radiance
            if not only_diffuse:
                self.two_bounce_args = two_bounce_cfg['nerf_cfg']
                embedding_xyz = Embedding(3, 10) # 10 is the default number
                embedding_dir = Embedding(3, 4) # 4 is the default number
                self.embeddings_model = [embedding_xyz, embedding_dir]
                # currently using default NeRF formulation  
                nerf_coarse = NeRF()
                self.nerf_models = [nerf_coarse]
                if two_bounce_cfg['nerf_cfg'].N_importance > 0:
                    nerf_fine = NeRF()
                    self.nerf_models += [nerf_fine]

        elif self.use_env_mlp == "env_as_5d_radiance_field_mip_nerf":
            # implements a 5d radiance field: with inputs (x, IDE) -> F -> (radiance, sigma) 
            # We integrate the ray to get radiance
            self.mip_nerf = MipNerf(args=two_bounce_cfg['mip_nerf_cfg'])
        
        elif self.use_env_mlp == "virtual_cone_lens_approx_5drf_mip_nerf":
            print("Instantiating MipNerf Module with virtual cone and lens approximation. {}".format(self.use_env_mlp))
            self.mip_nerf = MipNerf(args=two_bounce_cfg['mip_nerf_cfg'])
        
        elif self.use_env_mlp == "virtual_cone_caustic_approx_5drf_mip_nerf":
            print("Instantiating MipNerf Module with virtual cone and caustic approximation. {}".format(self.use_env_mlp))
            self.mip_nerf = MipNerf(args=two_bounce_cfg['mip_nerf_cfg'])
        else:
            self.diffuse_net = RadianceNet(
                                W_geo_feat=W_geo_feat, 
                                final_act='sigmoid',
                                **radiance_cfg)
            if not only_diffuse:
                self.specular_net = RadianceNet(
                                    W_geo_feat=W_geo_feat,  
                                    final_act='swish',
                                    **radiance_cfg)

        # save two_bounce_cfg
        self.two_bounce_cfg = two_bounce_cfg

        if use_nerfplusplus:
            self.nerf_outside = NeRF(input_ch=4, multires=10, multires_view=4, use_view_dirs=True)

    def forward_ab(self):
        beta = torch.exp(self.ln_beta * self.speed_factor)
        return 1./beta, beta

    def forward_surface(self, x: torch.Tensor):
        sdf = self.implicit_surface.forward(x)
        if self.use_sphere_bg:
            return torch.min(sdf, self.obj_bounding_radius - x.norm(dim=-1))
        else:
            return sdf        

    def forward_surface_with_nablas(self, x: torch.Tensor):
        sdf, nablas, h = self.implicit_surface.forward_with_nablas(x)
        if self.use_sphere_bg:
            d_bg = self.obj_bounding_radius - x.norm(dim=-1)
            # outside_sphere = x_norm >= 3
            outside_sphere = d_bg < sdf # NOTE: in case the normals changed suddenly near the sphere.
            sdf[outside_sphere] = d_bg[outside_sphere]
            # nabla[outside_sphere] = normals_bg_sphere[outside_sphere] # ? NOTE: commented to ensure more eikonal constraints. 
        return sdf, nablas, h

    def forward_with_nablas_curvature(self, x: torch.Tensor): 
        sdf, nablas, curvature, h = self.implicit_surface.forward_with_nablas_curvature(x)
        if self.use_sphere_bg:
            d_bg = self.obj_bounding_radius - x.norm(dim=-1)
            # outside_sphere = x_norm >= 3
            outside_sphere = d_bg < sdf # NOTE: in case the normals changed suddenly near the sphere.
            sdf[outside_sphere] = d_bg[outside_sphere]
            # nabla[outside_sphere] = normals_bg_sphere[outside_sphere] # ? NOTE: commented to ensure more eikonal constraints. 
        return sdf, nablas, curvature, h

    def forward_one_bounce(self, x:torch.Tensor, view_dirs: torch.Tensor, only_diffuse: torch.bool):
        # only_diffuse was broadcasted to pass through batchify
        # Convert it back to a scalar
        only_diffuse = only_diffuse.sum().bool()


        if self.use_env_mlp in ["rough_mask_envmap_MLP", "simple_env_5dscene_repres", \
                                 "env_as_5d_radiance_field_nerf", "env_as_5d_radiance_field_mip_nerf"]:
            sdf, nablas, geometry_feature = self.forward_surface_with_nablas(x)
        elif self.use_env_mlp in ["virtual_cone_lens_approx_5drf_mip_nerf", "virtual_cone_caustic_approx_5drf_mip_nerf"]:
            sdf, nablas, curvature, geometry_feature = self.forward_with_nablas_curvature(x)

        # Perform mask out: 
        mask_out = self.mask_net.forward(x, geometry_feature)
        # print("mask_out: {}".format(mask_out.shape))
        # Perform Diffuse out: 
        if self.diffuse_net is not None:
            diffnet_out = self.diffuse_net.forward(x, geometry_feature)
        else:
            diffnet_shape = [mask_out.shape[0], mask_out.shape[1], 3]
            diffnet_out = torch.zeros(diffnet_shape).to(mask_out.device)
            
        if self.use_env_mlp == "rough_mask_envmap_MLP":
            return diffnet_out, mask_out, sdf, nablas, geometry_feature
        elif self.use_env_mlp == "simple_env_5dscene_repres":
            return diffnet_out, mask_out, sdf, nablas, geometry_feature
        elif self.use_env_mlp == "env_as_5d_radiance_field_nerf":
            return diffnet_out, mask_out, sdf, nablas, geometry_feature
        elif self.use_env_mlp == "env_as_5d_radiance_field_mip_nerf":
            # phase II w/ roughenss (x, dir) -> F -> (radiance, sigma); We integrate the ray to get radiance
            return diffnet_out, mask_out, sdf, nablas, geometry_feature
        elif self.use_env_mlp == "virtual_cone_lens_approx_5drf_mip_nerf":
            derivatives = torch.cat([nablas, curvature.unsqueeze(-1)], dim=-1)
            return diffnet_out, mask_out, sdf, derivatives, geometry_feature
        elif self.use_env_mlp == "virtual_cone_caustic_approx_5drf_mip_nerf":
            derivatives = torch.cat([nablas, curvature.unsqueeze(-1)], dim=-1)
            return diffnet_out, mask_out, sdf, derivatives, geometry_feature
        else:
            diffnet_out = self.diffuse_net.forward(x, view_dirs, nablas, geometry_feature)
            return diffnet_out, sdf, nablas, geometry_feature

    def forward_two_bounce(self, x: torch.tensor, view_dirs: torch.Tensor, 
                               nablas:torch.Tensor, diffuse_radiance: torch.Tensor, 
                               mask_out: torch.Tensor, geometry_feature:torch.Tensor, 
                               near: float, far: float
                               ):
        """
        Outputs the specular Radiance: 
            x: Origin of the Primary Ray [1, -1, 3]
            depth: [N_rays, N_pts, 1] (N_pts could be 1)
            view_dirs: [1, -1, 3]
            depth: if depth is None then we use x as the origin of the secondary ray! 
        """
        # print("Inside Two Bounce with bounds ({},{})".format(near.max(), far.max()))

        if self.use_env_mlp == "rough_mask_envmap_MLP":
            if self.env_mlp_type == 'no_fres':
                diffuse_refl = diffuse_radiance[...,:3]
                normal_vecs = F.normalize(nablas, dim=-1)
                # print("inside ", normal_vecs.shape, view_dirs.shape)
                refl_vecs = view_dirs - (2 * torch.unsqueeze(torch.sum(view_dirs * normal_vecs, 2), 2) * normal_vecs) # refl = ray_d - 2 * dot(rays_d, normals) * normals
                # print("reflec_vec", torch.linalg.norm(refl_vecs,  dim=-1).max())
                # assert torch.linalg.norm(refl_vecs,  dim=-1).max() > 1 
                alphas_diffout =  self.rough_net.forward(x, geometry_feature)
                # TEMPORARY FIX FOR MATERIAL EDITING ROUGH APPEARANCE
                # alphas_diffout =  3.*self.rough_net.forward(x, geometry_feature)
                speculars = self.specular_net.forward(refl_vecs, alphas_diffout)
                radiances = torch.cat([diffuse_refl, speculars, alphas_diffout,mask_out],-1)
                return radiances
            elif self.env_mlp_type == 'fres_input':
                raise ValueError("I've disabled non-default pandora implementation...")

        if self.use_env_mlp == "simple_env_5dscene_repres":
            diffuse_refl = diffuse_radiance[...,:3]
            normal_vecs = F.normalize(nablas, dim=-1)
            refl_vecs = view_dirs - (2 * torch.unsqueeze(torch.sum(view_dirs * normal_vecs, 2), 2) * normal_vecs) # refl = ray_d - 2 * dot(rays_d, normals) * normals
            alphas_diffout =  self.rough_net.forward(x, geometry_feature)
            speculars = self.specular_net.forward(x, refl_vecs, alphas_diffout)
            radiances = torch.cat([diffuse_refl, speculars, alphas_diffout,mask_out],-1)
            return radiances

        ##### Two bounce Configuration #####
        if 'detach_nablas' in self.two_bounce_cfg:
            if self.two_bounce_cfg['detach_nablas']:
                nablas = nablas.detach()

        if self.sampling_type == "5dreflected_only" and \
            self.use_env_mlp != 'virtual_cone_caustic_approx_5drf_mip_nerf':
            normal_vecs = F.normalize(nablas, dim=-1)
            refl_vecs = view_dirs - (2 * torch.unsqueeze(torch.sum(view_dirs * normal_vecs, 2), 2) * normal_vecs) 
            refl_vecs = F.normalize(refl_vecs, dim=-1)
            # print("refl_vecs is normalized? ", refl_vecs.shape, torch.norm(refl_vecs, dim=-1))
            # print("normal_vecs is normalized? ", torch.norm(normal_vecs, dim=-1))

        if self.use_env_mlp == "env_as_5d_radiance_field_nerf":
            # We sample ray originating at xs with dir refl_vec
            result = render_rays(self.nerf_models, self.embeddings_model, 
                                 rays_o=x, rays_d=refl_vecs, 
                                 near = near, 
                                 far = far, 
                                 N_samples=self.two_bounce_args.num_samples, 
                                 use_disp=False, perturb=False, noise_std=1, 
                                 N_importance=self.two_bounce_args.N_importance, 
                                 chunk=1024*32, white_back=False, test_time=False)
            radiances = torch.cat([diffuse_radiance, result['rgb_fine'], mask_out],-1)
            return radiances, result
        elif self.use_env_mlp == "env_as_5d_radiance_field_mip_nerf" or \
              self.use_env_mlp == "virtual_cone_lens_approx_5drf_mip_nerf" or \
               self.use_env_mlp == "virtual_cone_caustic_approx_5drf_mip_nerf":
            rays = dotdict()
            rays.origins = x.reshape(-1, 3) # (B, N, 3) -> (B*N,3)

            if self.use_env_mlp != "virtual_cone_caustic_approx_5drf_mip_nerf":
                rays.directions = refl_vecs.reshape(-1, 3) # (B, N, 3) -> (B*N,3)
                rays.viewdirs = refl_vecs.reshape(-1, 3) # volsdf everything is normalized so should be the same 
                rays.radii = compute_radii(refl_vecs).reshape(-1, 1).to(rays.origins.device) #-> (B*N,1)
            else:
                virtual_refl_dirs = view_dirs[...,0:3].reshape(-1, 3)
                rays.directions = virtual_refl_dirs
                rays.viewdirs = virtual_refl_dirs
                # print("refl_vecs is normalized? ", virtual_refl_dirs.shape, torch.norm(virtual_refl_dirs, dim=-1))
                virtual_radii = view_dirs[...,3].reshape(-1, 1)
                rays.radii = virtual_radii

            # print("rays.radii", rays.radii.shape, rays.directions.shape)
            rays.near = near.reshape(rays.radii.shape)
            rays.far  = far.reshape(rays.radii.shape)

            # detach stuff 
            if 'detach_x' in self.two_bounce_cfg:
                if self.two_bounce_cfg['detach_x']:
                    rays.origins = rays.origins.detach()

            if 'detach_radii' in self.two_bounce_cfg:
                if self.two_bounce_cfg['detach_radii']:
                    rays.radii = rays.radii.detach()
            
            # should be already detached
            rays.near = rays.near.detach()
            rays.far  = rays.far.detach()
            
            ret = self.mip_nerf.forward(rays=rays, randomized=True, white_bkgd=False)
            c_rgb, c_distance, _, c_weights, c_t_samples = ret[0] # first pass is coarse sampling 
            f_rgb, f_distance, f_acc, f_weights, f_t_samples = ret[-1] # the last one is the finest sampling
            result = {}
            # todo (ktiwaery) fix this 1,-1,3 shape !!!!!! or else wont work with batchsize >1 
            result['rgb_coarse'] = c_rgb.reshape(diffuse_radiance.shape)
            result['rgb_fine'] = f_rgb.reshape(diffuse_radiance.shape)
            result['depth_coarse'] = c_distance.reshape(mask_out.shape)
            result['depth_fine'] = f_distance.reshape(mask_out.shape)
            # f_weights, f_t_samples: torch.Size([12288, 64]) torch.Size([12288, 65])
            # c_weights = c_weights.reshape((mask_out.shape[0],mask_out.shape[1], -1))
            # c_weights = c_weights.reshape((mask_out.shape[0],mask_out.shape[1], -1))

            dlosses = []
            for i in range(len(ret)):
                _, _, _, w, t = ret[i]
                #d = lossfun_distortion(w, t)
                d = distloss(w, t)
                dlosses.append(d)
            dloss_fine = dlosses[0] #.unsqueeze(0).unsqueeze(-1)
            dloss_coarse = dlosses[-1] #.unsqueeze(0).unsqueeze(-1)
            dloss_fine = torch.ones(mask_out.shape).to(mask_out.device) * dloss_fine
            dloss_coarse = torch.ones(mask_out.shape).to(mask_out.device) * dloss_coarse 

            radiances = torch.cat([diffuse_radiance, result['rgb_fine'], result['rgb_coarse'], \
                                    result['depth_coarse'], result['depth_fine'], mask_out, dloss_fine, dloss_coarse], -1)
            return radiances

        elif self.use_env_mlp == "virtual_cone_caustic_approx_5drf_mip_nerf": 
            raise NotImplementedError("virtual_cone_caustic_approx_5drf_mip_nerf not implemented...")

        else:
            speculars = self.specular_net.forward(x, view_dirs, nablas, geometry_feature)
            radiances = torch.cat([diffuse_radiance, speculars],-1)
            return radiances

# With help from Unreal Engine Notes: https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
# And Coding Labs Notes: http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx
def relight(v, n, newmap, rough):
    print(v.shape, n.shape, rough.shape)
    raise ValueError('Re-lighting is not ready yet')

def volume_render(
    rays_o, 
    rays_d,
    model: VolSDF,
    
    near=0.0,
    far=6.0,
    # (NEW): Explicitly set Mip nerf's near and far bounds
    twob_near=0.0,
    twob_far=6.0,
    obj_bounding_radius=3.0,
    
    batched = False,
    batched_info = {},
    
    # render algorithm config
    calc_normal = False,
    use_view_dirs = True,
    rayschunk = 65536,
    netchunk = 1048576,
    white_bkgd = False,
    use_nerfplusplus = False,
    
    # render function config
    detailed_output = True,
    show_progress = False,
    
    # sampling related
    perturb = False,   # config whether do stratified sampling
    N_samples = 128,
    N_importance = 64,
    N_outside = 32,
    max_upsample_steps = 5,
    max_bisection_steps = 10,
    epsilon = 0.1,

    use_env_mlp = "no_envmap_MLP",
    sh_embed = False,
    relight_map = None,

    # polarized rendering
    polarized = False,
    pol_render_type='stokes_accum',
    only_diffuse=False,

    # two bounce kwargs
    use_depth_for_two_bounce: bool = False,
    nan_to_num_enabled = True,
    rays_radii = None, 
    use_refl_as_vdirs = False,

    **dummy_kwargs  # just place holder
    ):
    """
    input: 
        rays_o: [(B,) N_rays, 3]
        rays_d: [(B,) N_rays, 3] NOTE: not normalized. contains info about ratio of len(this ray)/len(principle ray)
    """
    device = rays_o.device
    if batched:
        DIM_BATCHIFY = 1
        B = rays_d.shape[0]  # batch_size
        flat_vec_shape = [B, -1, 3]
    else:
        DIM_BATCHIFY = 0
        flat_vec_shape = [-1, 3]

    rays_o = torch.reshape(rays_o, flat_vec_shape).float()
    rays_d = torch.reshape(rays_d, flat_vec_shape).float()
    # NOTE: already normalized
    rays_d = F.normalize(rays_d, dim=-1)
    
    # pdb.set_trace()
    if use_env_mlp == 'virtual_cone_caustic_approx_5drf_mip_nerf':
        rays_d = torch.concat([rays_d, rays_radii.unsqueeze(-1)], dim=-1)

    
    batchify_query = functools.partial(train_util.batchify_query, chunk=netchunk, dim_batchify=DIM_BATCHIFY)
    # ---------------
    # Render a ray chunk
    # ---------------
    def render_rayschunk(rays_o: torch.Tensor, rays_d: torch.Tensor, 
                         only_diffuse=False):
        # rays_o: [(B), N_rays, 3]
        # rays_d: [(B), N_rays, 3]

        if use_env_mlp == 'virtual_cone_caustic_approx_5drf_mip_nerf':
            rays_radii = rays_d[..., 3]
            rays_d = rays_d[..., 0:3]

        if use_view_dirs:
            view_dirs = rays_d
        else:
            view_dirs = None
        
        prefix_batch = [B] if batched else []
        N_rays = rays_o.shape[-2]
        
        nears = near * torch.ones([*prefix_batch, N_rays, 1]).to(device)
        if use_nerfplusplus:
            _, fars, mask_intersect = rend_util.get_sphere_intersection(rays_o, rays_d, r=obj_bounding_radius)
            assert mask_intersect.all()
        else:
            fars = far * torch.ones([*prefix_batch, N_rays, 1]).to(device)

        # ---------------
        # Sample points on the rays
        # ---------------
        
        # ---------------
        # Coarse Points
        _t = torch.linspace(0, 1, N_samples).float().to(device)
        # [(B), N_rays, N_samples]
        d_coarse = nears * (1 - _t) + fars * _t
        
        # ---------------
        # Fine sampling algorithm
        alpha, beta = model.forward_ab()
        with torch.no_grad():
            # d_init = d_coarse
            
            # NOTE: setting denser d_init boost up up_sampling convergence without sacrificing much speed (since no grad here.)
            _t = torch.linspace(0, 1, N_samples*4).float().to(device) # NOTE: you might want to use less samples for faster training.
            d_init = nears * (1 - _t) + fars * _t
            
            d_fine, beta_map, iter_usage = fine_sample(
                model.forward_surface, d_init, rays_o, rays_d, 
                alpha_net=alpha, beta_net=beta, far=fars, 
                eps=epsilon, max_iter=max_upsample_steps, max_bisection=max_bisection_steps, 
                final_N_importance=N_importance, perturb=perturb, 
                N_up=N_samples*4    # NOTE: you might want to use less samples for faster training.
            )

        # ---------------
        # Gather points
        # NOTE: from the paper, should not concatenate here; 
        # NOTE: but from practice, as long as not concatenating and only using fine points, 
        #       there would be artifact emerging very fast before 10k iters, and the network converged to severe local minima (all cameras inside surface).
        d_all = torch.cat([d_coarse, d_fine], dim=-1)
        d_all, sort_indices = torch.sort(d_all, dim=-1)
        if model.fine_spec:
            fine_mask = torch.cat([ torch.zeros(d_coarse.shape),
                                    torch.ones(d_fine.shape)],dim=-1).float().to(device)
            fine_mask_sorted = torch.gather(fine_mask,-1,sort_indices)# [(B), N_rays, N_pts]
        # d_all = d_fine
        #  torch.Size([1, 256, 3]) torch.Size([1, 256, 192]) torch.Size([1, 256, 3])
        pts = rays_o[..., None, :] + rays_d[..., None, :] * d_all[..., :, None]
        # pts torch.Size([1, 256, 192, 3])
        
        # ---------------
        # Qeury network
        # pts.shape, view_dirs.shape, view_dirs.unsqueeze(-2).expand_as(pts).shape
        # [(B), N_rays, N_pts, 3],   # [(B), N_rays, N_pts]   [(B), N_rays, N_pts, W_geo]
        view_dirs_pts = view_dirs.unsqueeze(-2).expand_as(pts) if use_view_dirs else None

        # initialize distlosses in case its not set
        dloss_fine = None
        dloss_coarse = None

        if use_env_mlp not in ["virtual_cone_lens_approx_5drf_mip_nerf", "virtual_cone_caustic_approx_5drf_mip_nerf"]:
            diffnet_out, mask_out, sdf, \
                nablas, geometry_feature = batchify_query(model.forward_one_bounce, pts, 
                                                    view_dirs_pts,
                                                    only_diffuse+torch.zeros(pts.shape, dtype=torch.bool,device=pts.device))
        elif use_env_mlp in ["virtual_cone_lens_approx_5drf_mip_nerf", "virtual_cone_caustic_approx_5drf_mip_nerf"]:
            diffnet_out, mask_out, sdf, derivatives, \
                 geometry_feature = batchify_query(model.forward_one_bounce, pts, 
                                                    view_dirs_pts,
                                                    only_diffuse+torch.zeros(pts.shape, dtype=torch.bool,device=pts.device))
            nablas = derivatives[..., 0:3]
            curvature = derivatives[..., 3]
            assert not torch.isnan(nablas).any()
            assert not torch.isnan(curvature).any()
        else:
            raise NotImplementedError("{}'s single bounce not implemented.".format(use_env_mlp))
            
        flag_use_coarse_specular = False
        if use_env_mlp in ['rough_mask_envmap_MLP']:
            if not only_diffuse:
                radiances = batchify_query(model.forward_two_bounce, pts, view_dirs_pts, 
                                              nablas, diffnet_out, mask_out, geometry_feature, 
                                              torch.ones(pts.unsqueeze(3).shape[:-1], device=pts.device) * twob_near, 
                                              torch.ones(pts.unsqueeze(3).shape[:-1], device=pts.device) * twob_far
                                              )
            else:
                radiances = torch.cat([diffnet_out, mask_out],-1)
   
        elif use_env_mlp in ['simple_env_5dscene_repres']:
            if not only_diffuse:
                pts_2b = pts
                view_dirs_pts_2b = view_dirs_pts
                radiances = batchify_query(model.forward_two_bounce, pts_2b, view_dirs_pts_2b, 
                                              nablas, diffnet_out, mask_out, geometry_feature, 
                                            #   nablas.detach(), diffnet_out, mask_out, geometry_feature, 
                                              torch.ones(pts.unsqueeze(3).shape[:-1], device=pts.device) * twob_near, 
                                              torch.ones(pts.unsqueeze(3).shape[:-1], device=pts.device) * twob_far)
            else:
                radiances = torch.cat([diffnet_out, mask_out],-1)

        elif use_env_mlp in ['env_as_5d_radiance_field_nerf', 'env_as_5d_radiance_field_mip_nerf']:
            if not only_diffuse:
                flag_use_coarse_specular = True
                radiances = batchify_query(model.forward_two_bounce, pts, view_dirs_pts, 
                                               nablas, diffnet_out, mask_out, geometry_feature, 
                                               torch.ones(pts.unsqueeze(3).shape[:-1], device=pts.device) * twob_near, 
                                               torch.ones(pts.unsqueeze(3).shape[:-1], device=pts.device) * twob_far)
                coarse_2bounce_distance = radiances[..., :, -5].reshape(-1, 1)
                fine_2bounce_distance = radiances[..., :, -4].reshape(-1, 1)
                dloss_fine = radiances[..., :, -2].reshape(-1, 1)
                dloss_coarse = radiances[..., :, -1].reshape(-1, 1)
                radiances = radiances[:,:,:,:-2]
            else:
                radiances = torch.cat([diffnet_out, mask_out],-1)
        elif use_env_mlp in ['virtual_cone_lens_approx_5drf_mip_nerf']:
            if not only_diffuse:
                flag_use_coarse_specular = True
                # print('virtual_cone_lens_approx_5drf_mip_nerf', pts.shape, view_dirs_pts.shape, 
                #       derivatives.shape, diffnet_out.shape, mask_out.shape, geometry_feature.shape)
                # virtual_cone_lens_approx_5drf_mip_nerf torch.Size([1, 64, 192, 3]) torch.Size([1, 64, 192, 3]) 
                        # torch.Size([1, 64, 192, 3]) torch.Size([1, 64, 192, 3]) torch.Size([1, 64, 192, 1]) torch.Size([1, 64, 192, 256])
                # calculate virtual points using Lens Approximation
                virutal_pts = get_virtual_viewpoint_lens_approx(derivatives, view_dirs_pts, pts, d_all, 
                                                                    nan_to_num_enabled = nan_to_num_enabled)
                radiances = batchify_query(model.forward_two_bounce, virutal_pts, view_dirs_pts, 
                                              nablas, diffnet_out, mask_out, geometry_feature, 
                                              torch.ones(pts.unsqueeze(3).shape[:-1], device=pts.device) * twob_near, 
                                              torch.ones(pts.unsqueeze(3).shape[:-1], device=pts.device) * twob_far)
                coarse_2bounce_distance = radiances[..., :, -5].reshape(-1, 1)
                fine_2bounce_distance = radiances[..., :, -4].reshape(-1, 1)
                dloss_fine = radiances[..., :, -2].reshape(-1, 1)
                dloss_coarse = radiances[..., :, -1].reshape(-1, 1)
                radiances = radiances[:,:,:,:-2]
            else:
                radiances = torch.cat([diffnet_out, mask_out],-1)
                # print("radiances", radiances.shape)
        elif use_env_mlp in ['virtual_cone_caustic_approx_5drf_mip_nerf']:
            if not only_diffuse:
                flag_use_coarse_specular = True
                # flags 
                DETACH_CURVATURE = True
                DETACH_RADII = True
                COMPUTED_NEAR = False
                # o,d,r,up,t,n,R
                # pdb.set_trace()
                origins = rays_o.unsqueeze(2).expand(pts.shape)
                up_vec = torch.tensor([0.,-1.,0.]).float() # see two_bounce_utils:336
                up_pts = up_vec.expand(pts.shape).to(pts.device)
                normal_vecs = F.normalize(nablas, dim=-1)
                # normal_vecs = nablas
                if DETACH_CURVATURE:
                    curvature = curvature.detach() # [N_rays, N_pts]
                radius_of_curvature = 2./(curvature + 1.e-7) # [B, N_rays, N_pts]
                radius_of_curvature = radius_of_curvature.reshape(pts.unsqueeze(3).shape[:-1])
                radii = rays_radii.unsqueeze(-1).unsqueeze(-1).expand(pts.unsqueeze(3).shape[:-1])
                if DETACH_RADII:
                    radii = radii.detach()

                try:
                    virtual_pts, virtual_dirs, virtual_radii = get_virtual_cone_params_sphere(
                                                                        o=origins.squeeze(0), 
                                                                        d=view_dirs_pts.squeeze(0),
                                                                        r=radii.squeeze(0), 
                                                                        up=up_pts.squeeze(0), 
                                                                        t=d_all.detach().squeeze(0).unsqueeze(-1) + 1.0e-3, 
                                                                        n=normal_vecs.squeeze(0), 
                                                                        R=radius_of_curvature.squeeze(0))
                    # import pdb 
                    # pdb.set_trace()
                except Exception as e:
                    dict_ = {}
                    dict_['o'] = origins.squeeze(0)
                    dict_['d'] = view_dirs_pts.squeeze(0)
                    dict_['r'] = radii.squeeze(0)
                    dict_['up'] = up_pts.squeeze(0)
                    dict_['t'] = d_all.squeeze(0).unsqueeze(-1) + 1.0e-3
                    dict_['n'] = normal_vecs.squeeze(0)
                    dict_['R'] = radius_of_curvature.squeeze(0)
                    import time
                    # fname = 'found_an_error_with_these_inputs_{}.pt'.format(time.time())
                    fname = 'viz/test_inputs.pt'.format(time.time())
                    torch.save(dict_, fname)
                    print("NOTE: Virtual Inputs saved state at {}".format(fname))
                    import pdb; pdb.set_trace()
                    raise e
                
                if use_refl_as_vdirs:
                    refl_vecs = view_dirs_pts - \
                        (2 * torch.sum(view_dirs_pts * normal_vecs, -1, keepdim=True) * normal_vecs)
                    refl_vecs = F.normalize(refl_vecs, dim=-1)
                    virtual_bundle = torch.cat([refl_vecs.squeeze(0), virtual_radii.detach()], dim=-1)
                else:
                    virtual_bundle = torch.cat([virtual_dirs, virtual_radii], dim=-1).detach()
                twob_far_expand = torch.ones(pts.unsqueeze(3).shape[:-1], device=pts.device) * twob_far
                twob_near_expand_inp = torch.ones(pts.unsqueeze(3).shape[:-1], device=pts.device) * twob_near
                if COMPUTED_NEAR:
                    twob_near_expand_comp = ((pts - virtual_pts)**2).sum(-1, keepdim=True).sqrt()
                    twob_near_expand = torch.maximum(twob_near_expand_comp, twob_near_expand_inp).detach()
                    max_clip = torch.minimum(0.9*twob_far_expand, 0.5*radius_of_curvature)
                    # max_clip = torch.minimum(0.5*twob_far_expand, 0.5*radius_of_curvature)
                    twob_near_expand = torch.minimum(max_clip, twob_near_expand)
                else:
                    twob_near_expand = twob_near_expand_inp
                radiances = batchify_query(model.forward_two_bounce, virtual_pts.unsqueeze(0), 
                                             virtual_bundle.unsqueeze(0), 
                                             nablas, diffnet_out, mask_out, geometry_feature, 
                                             twob_near_expand, 
                                             twob_far_expand)
                # if True:
                if torch.isnan(radiances).any():
                    dict_ = {}
                    # dict_['twob_near'] = twob_near_expand.squeeze(0)
                    # dict_['twob_far'] = twob_far_expand.squeeze(0)
                    dict_['o'] = origins.squeeze(0)
                    dict_['d'] = view_dirs_pts.squeeze(0)
                    dict_['r'] = radii.squeeze(0)
                    dict_['up'] = up_pts.squeeze(0)
                    dict_['t'] = d_all.squeeze(0).unsqueeze(-1) + 1.0e-3
                    dict_['n'] = normal_vecs.squeeze(0)
                    dict_['R'] = radius_of_curvature.squeeze(0)
                    import time
                    fname = 'found_nans_with_these_inputs_{}.pt'.format(time.time())
                    # fname = 'good_vals_{}.pt'.format('living_room')
                    torch.save(dict_, fname)
                    print("NOTE: Virtual Inputs saved state at {}".format(fname))
                    print("saved state at {}".format(fname))
                    print("normal_vecs and curvature", torch.isnan(normal_vecs).any(), torch.isnan(curvature).any())
                    print("is nan:", torch.isnan(virtual_pts).any(),torch.isnan(virtual_dirs).any(), torch.isnan(virtual_radii).any())
                    raise ValueError("Nans found in radiances...")

                coarse_2bounce_distance = radiances[..., :, -5].reshape(-1, 1)
                fine_2bounce_distance = radiances[..., :, -4].reshape(-1, 1)
                dloss_fine = radiances[..., :, -2].reshape(-1, 1)
                dloss_coarse = radiances[..., :, -1].reshape(-1, 1)
                radiances = radiances[:,:,:,:-2]           
            else:
                radiances = torch.cat([diffnet_out, mask_out],-1)
        else: 
            raise NotImplementedError("{} not implemented...".format(use_env_mlp))

        # [(B), N_rays, N_pts]
        sigma = sdf_to_sigma(sdf, alpha, beta)

        # ---------------
        # NeRF++
        if use_nerfplusplus:
            raise NotImplementedError("nerf plus plus isn't implemented anymore...")            
        # ---------------
        # Ray integration
        # ---------------
        # [(B), N_rays, N_pts-1]
        # delta_i = (d_all[..., 1:] - d_all[..., :-1]) * rays_d.norm(dim=-1)[..., None]
        delta_i = d_all[..., 1:] - d_all[..., :-1]  # NOTE: aleardy real depth
        # [(B), N_rays, N_pts-1]
        p_i = torch.exp(-F.relu_(sigma[..., :-1] * delta_i))
        # [(B), N_rays, N_pts-1]
        # (1-p_i) * \prod_{j=1}^{i-1} p_j
        # NOTE: NOT (1-pi) * torch.cumprod(p_i)! the cumprod later should use shifted p_i! 
        #       because the cumprod ends to (i-1), not i.
        tau_i = (1 - p_i + 1e-10) * (
            torch.cumprod(
                torch.cat(
                    [torch.ones([*p_i.shape[:-1], 1], device=device), p_i], dim=-1), 
                dim=-1)[..., :-1]
            )
        if model.fine_spec:
            # [(B), N_rays, N_pts-1]
            p_i_fine = torch.exp(-F.relu_(fine_mask_sorted[...,:-1]*sigma[..., :-1] * delta_i))
            # [(B), N_rays, N_pts-1]
            tau_i_fine = (1 - p_i_fine + 1e-10) * (
                            torch.cumprod(
                                torch.cat(
                                    [torch.ones([*p_i_fine.shape[:-1], 1], device=device), p_i_fine], dim=-1), 
                                dim=-1)[..., :-1])

        # [(B), N_rays, 1]
        # print("d_all", d_all[..., :-1].shape)
        depth_map = torch.sum(tau_i / (tau_i.sum(-1, keepdim=True)+1e-10) * d_all[..., :-1], dim=-1)
        acc_map = torch.sum(tau_i, -1)
        # print("depth_map", depth_map.shape)

        if calc_normal or model.use_env_mlp != "no_envmap_MLP":
            normals_map = F.normalize(nablas, dim=-1)
            N_pts = min(tau_i.shape[-1], normals_map.shape[-2])
            normals_map = (normals_map[..., :N_pts, :] * tau_i[..., :N_pts, None]).sum(dim=-2)
            # print("normals_map", torch.isnan(normals_map).any())
            # normals_map = (fine_mask_sorted[...,:N_pts,None] * normals_map[..., :N_pts, :] * tau_i[..., :N_pts, None]).sum(dim=-2)

        # [(B), N_rays, 3]
        rgb_map = torch.sum(tau_i[..., None] * diff_net_to_lin(radiances[..., :-1, :3]), dim=-2)
        # [(B), N_rays, 3]
        if not only_diffuse:
            # print("specular amap all : {}".format(radiances[..., :-1, 3:6].shape))
            spec_map = torch.sum(tau_i[..., None] * spec_net_to_lin(radiances[..., :-1, 3:6]), dim=-2)
            # print("spec_shape other location", spec_map.shape)
            if flag_use_coarse_specular:
                # print(coarse_2bounce_spec_radiance[..., :-1, 0:3].shape)
                spec_map_coarse = torch.sum(tau_i[..., None] * \
                        spec_net_to_lin(radiances[..., :-1, 6:9]), dim=-2)
                # print("rgb_coarse", spec_map_coarse.shape)

        if model.use_env_mlp in ['rough_mask_envmap_MLP', 'simple_env_5dscene_repres', \
            'env_as_5d_radiance_field_nerf', 'env_as_5d_radiance_field_mip_nerf', \
            'virtual_cone_lens_approx_5drf_mip_nerf', 'virtual_cone_caustic_approx_5drf_mip_nerf']:
            # print("what it should be: ", radiances[...,:-1,-1].shape)
            # mask_map = torch.sum(tau_i * radiances[...,:-1,-1], 0, 1, dim=-1)
            mask_map = torch.sum(tau_i * radiances[...,:-1,-1], dim=-1)

        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        ret_i = OrderedDict([
            ('rgb', rgb_map),                # [(B), N_rays, 3]
            ('depth_volume', depth_map),     # [(B), N_rays]
            ('mask_volume', acc_map)            # [(B), N_rays]
        ])
        
        if model.use_env_mlp in ['virtual_cone_lens_approx_5drf_mip_nerf',
                                 'virtual_cone_caustic_approx_5drf_mip_nerf']:
            ret_i['implicit_curvature'] = curvature # [B, N_rays, N_pts, 1]
            ret_i['curvature_map'] = torch.sum(tau_i * curvature.reshape(radiances[...,:,-1].shape)[:,:,:-1], dim=-1) # [B, N_rays, 1]
        
        if not only_diffuse and model.use_env_mlp in ['virtual_cone_caustic_approx_5drf_mip_nerf']:
            N_pts = min(tau_i.shape[-1],virtual_pts.shape[-2])
            ret_i['v_up_map'] = (up_pts[...,:N_pts,:]*\
                                        tau_i[...,:N_pts,None]).sum(dim=-2)
            ret_i['v_pts_map'] = (virtual_pts[None,...,:N_pts,:]*\
                                        tau_i[...,:N_pts,None]).sum(dim=-2)
            ret_i['v_dirs_map'] = (virtual_dirs[None,...,:N_pts,:]*\
                                          tau_i[...,:N_pts,None]).sum(dim=-2)
            ret_i['v_radii_map'] = (virtual_radii[None,...,:N_pts,0]*\
                                          tau_i[...,:N_pts]).sum(dim=-1)

            
        if model.use_env_mlp in \
            ['env_as_5d_radiance_field_nerf', 'env_as_5d_radiance_field_mip_nerf', 
             'virtual_cone_lens_approx_5drf_mip_nerf','virtual_cone_caustic_approx_5drf_mip_nerf']:
            # raise 
            # print("coarse_2bounce_distance", coarse_2bounce_distance[:-1,:].shape, radiances[...,:-1,6].shape)
            # print("NEW", coarse_2bounce_distance.reshape(radiances[...,:,6].shape)[:,:,:-1].shape)
            if (not only_diffuse):
                # print("radiances[...,:,6].shape", radiances[...,:,6].shape)
                ret_i['fine_2bounce_depth'] = torch.sum(tau_i * fine_2bounce_distance.reshape(radiances[...,:,6].shape)[:,:,:-1], dim=-1)
                if flag_use_coarse_specular:
                    ret_i['coarse_2bounce_depth'] = torch.sum(tau_i * coarse_2bounce_distance.reshape(radiances[...,:,6].shape)[:,:,:-1], dim=-1)

        # removing it here and calculating 
        if calc_normal or model.use_env_mlp != "no_envmap_MLP":
            ret_i['normals_volume'] = normals_map

        if not only_diffuse and use_env_mlp in ['rough_mask_envmap_MLP', 'simple_env_5dscene_repres']:
            rough_map = torch.sum(tau_i * radiances[...,:-1,6], dim=-1)
        else:
            rough_map = None

        if relight_map == None:
            # spec_map = torch.sum(tau_i[..., None] * spec_net_to_lin(radiances[..., :-1, 3:6]), dim=-2)
            # print("spec_map inside relight", spec_map.shape)
            # spec_map = torch.sum(tau_i[..., None] * radiances[..., :-1, 3:], dim=-2)
            if not only_diffuse and not polarized:
                fresnel_refl = get_fresnel(rays_d, normals_map)
                spec_map *= fresnel_refl
                if flag_use_coarse_specular:
                    spec_map_coarse *= fresnel_refl
        else:
            relight(rays_d, normals_map, relight_map, rough_map)

        if polarized:
            # with torch.no_grad():
            normals_map = F.normalize(nablas, dim=-1)
            N_pts = min(tau_i.shape[-1], normals_map.shape[-2])
            if pol_render_type == 'stokes_accum':
                # 1: Accumulate normals
                normals_map = (normals_map[..., :N_pts, :] * tau_i[..., :N_pts, None]).sum(dim=-2)
                # 2: Compute Stokes vector from accumulated rgb, normals
                if only_diffuse:
                    stokes_sum_map_fine = stokes_from_normal_rad(rays_o, rays_d, normals_map, 
                                                        (rgb_map),train_mode=True)
                                                        #[(B), N_rays, 3(RGB), 3(s)]
                else:
                    diff_stokes_sum_map, spec_stokes_sum_map_fine = stokes_from_normal_rad(rays_o, rays_d, 
                                                        normals_map, 
                                                        rgb_map,
                                                        spec_rads=spec_map,
                                                        train_mode=True,
                                                        ret_separate=True)
                    if flag_use_coarse_specular:
                        _, spec_stokes_sum_map_coarse = stokes_from_normal_rad(rays_o, rays_d, 
                                                        normals_map, 
                                                        rgb_map,
                                                        spec_rads=spec_map_coarse,
                                                        train_mode=True,
                                                        ret_separate=True)
                        #[(B), N_rays, 3(RGB), 3(s)]
                        stokes_sum_map_coarse = diff_stokes_sum_map + spec_stokes_sum_map_coarse

                    stokes_sum_map_fine = diff_stokes_sum_map + spec_stokes_sum_map_fine

            elif pol_render_type == 'accum_stokes':
                # 1: Compute Stokes vector per point
                # [(B), N_rays, N_pts-1, 3(RGB), 3(s)]
                # with torch.no_grad():
                clip_spec = model.use_env_mlp != 'no_envmap_MLP'
                # clip_spec = False
                stokes_diff_fac_i, stokes_spec_fac_i, stokes_spec_fac0_i = stokes_fac_from_normal(rays_o[...,None],rays_d[...,None,:],
                                                                              normals_map[...,:N_pts,:],
                                                                              ret_spec=True,
                                                                              clip_spec=clip_spec) 

                diff_rads_i = diff_net_to_lin(radiances[...,:N_pts,:3])[...,None,:,None]
                # Apply mask from mask_net 
                if (model.use_env_mlp in ['rough_mask_envmap_MLP', 'simple_env_5dscene_repres', \
                    'env_as_5d_radiance_field_nerf', 'env_as_5d_radiance_field_mip_nerf', \
                    'virtual_cone_lens_approx_5drf_mip_nerf', 'virtual_cone_caustic_approx_5drf_mip_nerf']):
                    mask_i = radiances[...,:N_pts,[-1]]
                    diff_rads_i = diff_rads_i*mask_i[...,None,:,None]

                diff_stokes_i = (diff_rads_i*stokes_diff_fac_i).sum(-3)
                diff_stokes_sum_map = (diff_stokes_i[...,:N_pts,:,:]*tau_i[...,:N_pts,None,None]).sum(dim=-3)
                
                ## create this into a function and call it on coarse and fine? 
                if not only_diffuse:
                    # Apply env map output
                    spec_rads_i_fine = spec_net_to_lin(radiances[...,:N_pts,3:6])[...,None,:,None]
                    # print("spec_rads_i_fine", torch.isnan(spec_rads_i_fine).any())
                    if flag_use_coarse_specular:
                        spec_rads_i_coarse = spec_net_to_lin(radiances[...,:N_pts,6:9]).reshape(spec_rads_i_fine.shape)
                    # Apply mask from mask_net 
                    if (model.use_env_mlp in ['rough_mask_envmap_MLP', 'simple_env_5dscene_repres', \
                        'env_as_5d_radiance_field_nerf', 'env_as_5d_radiance_field_mip_nerf', \
                        'virtual_cone_lens_approx_5drf_mip_nerf', 'virtual_cone_caustic_approx_5drf_mip_nerf']):
                        mask_i = radiances[...,:N_pts,[-1]]
                        spec_rads_i_fine = spec_rads_i_fine*mask_i[...,None,:,None]
                        if flag_use_coarse_specular:
                            spec_rads_i_coarse = spec_rads_i_coarse*mask_i[...,None,:,None]
                            spec_stokes_i_coarse = (spec_rads_i_coarse*stokes_spec_fac_i).sum(-3)

                    spec_stokes_i_fine = (spec_rads_i_fine*stokes_spec_fac_i).sum(-3)
                    # Apply Fresnel reflectance
                    if (model.use_env_mlp != 'no_envmap_MLP') and model.env_mlp_type == 'fres_mlp':
                        raise ValueError("Not Implemented Error.")

                    if (model.use_env_mlp != 'no_envmap_MLP') and model.env_mlp_type == 'no_fres'\
                            and (not model.disable_fres):
                        spec_stokes_i_fine = spec_stokes_i_fine*(stokes_spec_fac0_i.sum(-3))
                        if flag_use_coarse_specular:
                            spec_stokes_i_coarse = spec_stokes_i_coarse*(stokes_spec_fac0_i.sum(-3))
                            # print("spec_stokes_i_coarse", torch.isnan(spec_stokes_i_coarse).any())

                    spec_stokes_sum_map_fine = (spec_stokes_i_fine[...,:N_pts,:,:]*tau_i[...,:N_pts,None,None]).sum(dim=-3)
                    if flag_use_coarse_specular:
                        spec_stokes_sum_map_coarse = (spec_stokes_i_coarse[...,:N_pts,:,:]*tau_i[...,:N_pts,None,None]).sum(dim=-3)
                    
                    spec_fac0_sum_map = ((stokes_spec_fac0_i.sum(-3))[...,:N_pts,:,:]*tau_i[...,:N_pts,None,None]).sum(dim=-3)

                if only_diffuse:
                    # stokes_sum_map = diff_stokes_sum_map
                    stokes_i = (diff_stokes_i)
                    stokes_sum_map_fine = (stokes_i[...,:N_pts,:,:]*tau_i[...,:N_pts,None,None]).sum(dim=-3)
                else:
                    # stokes_sum_map = diff_stokes_sum_map + spec_stokes_sum_map
                    stokes_i_fine = (diff_stokes_i+spec_stokes_i_fine)
                    stokes_sum_map_fine = (stokes_i_fine[...,:N_pts,:,:]*tau_i[...,:N_pts,None,None]).sum(dim=-3)
                    if flag_use_coarse_specular:
                        stokes_i_coarse = (diff_stokes_i+spec_stokes_i_coarse)
                        stokes_sum_map_coarse = (stokes_i_coarse[...,:N_pts,:,:]*tau_i[...,:N_pts,None,None]).sum(dim=-3)

                if  model.fine_spec:
                    normals_map = (normals_map[..., :N_pts, :] * tau_i_fine[..., :N_pts, None]).sum(dim=-2)
                else:
                    normals_map = (normals_map[..., :N_pts, :] * tau_i[..., :N_pts, None]).sum(dim=-2)
                # [(B), N_rays, 3(RGB), 3(s)]
            else:
                raise Exception(f'Invalid polarized rendering type {pol_render_type}')
            ret_i['normals_volume'] = normals_map
            # print("here normals", normals_map.shape)
            ret_i['s0'] = stokes_sum_map_fine[...,0]
            ret_i['s1'] = stokes_sum_map_fine[...,1]
            ret_i['s2'] = stokes_sum_map_fine[...,2]

            if flag_use_coarse_specular:
                # print("USING COARSE!!")
                ret_i['s0_coarse'] = stokes_sum_map_coarse[...,0]
                ret_i['s1_coarse'] = stokes_sum_map_coarse[...,1]
                ret_i['s2_coarse'] = stokes_sum_map_coarse[...,2]

        if polarized:
            if not only_diffuse:
                # set diffuse s0
                ret_i['diff_s0'] = diff_stokes_sum_map[...,0]
                ret_i['diff_s1'] = diff_stokes_sum_map[...,1]
                ret_i['diff_s2'] = diff_stokes_sum_map[...,2]
                # set specular s0
                ret_i['spec_map'] = spec_stokes_sum_map_fine[...,0]
                ret_i['spec_fac0'] = spec_fac0_sum_map[...,0]
                ret_i['spec_s0'] = spec_stokes_sum_map_fine[...,0]
                ret_i['spec_s1'] = spec_stokes_sum_map_fine[...,1]
                ret_i['spec_s2'] = spec_stokes_sum_map_fine[...,2]
                if flag_use_coarse_specular:
                    ret_i['spec_map_coarse'] = spec_stokes_sum_map_coarse[...,0]
                    ret_i['spec_s0_coarse'] = spec_stokes_sum_map_coarse[...,0]
                    ret_i['spec_s1_coarse'] = spec_stokes_sum_map_coarse[...,1]
                    ret_i['spec_s2_coarse'] = spec_stokes_sum_map_coarse[...,2]
        else:
            if not only_diffuse:
                ret_i['spec_map'] = spec_map

        if not only_diffuse:
            if use_env_mlp in ['rough_envmap_MLP','rough_mask_envmap_MLP', 'simple_env_5dscene_repres']:
                ret_i['rough_map'] = rough_map

        if use_env_mlp in ['rough_mask_envmap_MLP', 'simple_env_5dscene_repres', 'env_as_5d_radiance_field_nerf', \
                    'env_as_5d_radiance_field_mip_nerf', \
                    'virtual_cone_lens_approx_5drf_mip_nerf', 'virtual_cone_caustic_approx_5drf_mip_nerf']:
            ret_i['mask_map'] = mask_map

        # Calculate diffuse albedo
        if not only_diffuse and model.use_env_mlp \
            in ['rough_envmap_MLP','rough_mask_envmap_MLP'] \
                and model.env_mlp_type=='no_fres':
            r = 1.0 
            r_torch = r * torch.ones(normals_map.shape[:-1], dtype=normals_map.dtype, device=normals_map.device)
            ret_i['albedo'] = rgb_map / model.specular_net(normals_map, r_torch)
        else:
            ret_i['albedo'] = rgb_map

        # TODO (ktiwary): turned off atm
        # if not only_diffuse and (use_env_mlp in ['env_as_5d_radiance_field_mip_nerf', \
        #         'virtual_cone_lens_approx_5drf_mip_nerf', 'virtual_cone_caustic_approx_5drf_mip_nerf']):
        #     # CURRENTLY HARD CODED TO RUN WITH 64 SAMPLES ON THE SECONDARY RAY!!!! 
        #     ret_i['f_weights'] = radiances[...,:,9:73].reshape(-1, 64) #(64+9) see forward_two_bounce for infor
        #     ret_i['f_t_samples'] = radiances[...,:,73:138].reshape(-1, 65) # radiances[...,:, (64+9):(64+9+65)]

        ret_i['implicit_nablas'] = nablas
        # ret_i['radiance'] = radiances
        ret_i['d_vals'] = d_all
        if detailed_output:
            # [(B), N_rays, N_pts, ]
            ret_i['implicit_surface'] = sdf
            ret_i['alpha'] = 1.0 - p_i
            ret_i['p_i'] = p_i
            ret_i['visibility_weights'] = tau_i
            ret_i['sigma'] = sigma
            # [(B), N_rays, ]
            ret_i['beta_map'] = beta_map
            ret_i['iter_usage'] = iter_usage
            if use_nerfplusplus:
                ret_i['sigma_out'] = sigma_out
                ret_i['radiance_out'] = radiance_out
        if dloss_fine is not None and dloss_coarse is not None:
            ret_i['dloss_fine'] = dloss_fine
            ret_i['dloss_coarse'] = dloss_coarse

        return ret_i
        
    ret = {}
    # print("DIM_BATCHIFY", DIM_BATCHIFY)
    for i in tqdm(range(0, rays_o.shape[DIM_BATCHIFY], rayschunk), disable=not show_progress):
        ret_i = render_rayschunk(
            rays_o[:, i:i+rayschunk] if batched else rays_o[i:i+rayschunk],
            rays_d[:, i:i+rayschunk] if batched else rays_d[i:i+rayschunk],
            only_diffuse=only_diffuse,
        )
        for k, v in ret_i.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)
    for k, v in ret.items():
        # print("k:", k, len(v))
        if "dloss" in k:
            ret[k] = torch.tensor([t[0] for t in v]).to(rays_o.device).unsqueeze(0)
        else:
            ret[k] = torch.cat(v, DIM_BATCHIFY)
    
    # # NOTE: this is for debugging, which maintains computation graph. But not suitable for validation
    # ret = render_rayschunk(rays_o, rays_d)
    return ret['rgb'], ret['depth_volume'], ret


def volume_render_two_bounce(
             twob_model,
             rays_o, 
             rays_d, 
             near, # should be twob_near
             far, # should be twob_far
             batched = False,
            # render algorithm config
             rayschunk = 2048,
             netchunk = 1048576,
             randomized = True,
             white_bkgd = False,
            # render function config
             show_progress = False,
):
    device = rays_o.device
    if batched:
        DIM_BATCHIFY = 1
        B = rays_d.shape[0]  # batch_size
        flat_vec_shape = [B, -1, 3]
    else:
        DIM_BATCHIFY = 0
        flat_vec_shape = [-1, 3]

    batchify_query = functools.partial(train_util.batchify_query, chunk=netchunk, dim_batchify=DIM_BATCHIFY)

    def _batched_forward(x, refl_vecs, nears, fars):
        rays = dotdict()
        rays.origins = x.reshape(-1, 3) # (B, N, 3) -> (B*N,3)
        rays.directions = refl_vecs.reshape(-1, 3) # (B, N, 3) -> (B*N,3)
        rays.viewdirs = refl_vecs.reshape(-1, 3) # volsdf everything is normalized so should be the same 
        rays.radii = compute_radii(refl_vecs.reshape(1,-1,3)).reshape(-1, 1).to(rays.origins.device) #-> (B*N,1)
        # print("rays.radii", rays.radii)
        rays.near = nears.reshape(rays.radii.shape)
        rays.far  = fars.reshape(rays.radii.shape)

        ret = twob_model.forward(rays=rays, randomized=randomized, white_bkgd=white_bkgd)
        c_rgb, c_distance, _, _, _ = ret[0] # first pass is coarse sampling 
        f_rgb, f_distance, f_acc, f_weights, f_t_samples = ret[-1] # the last one is the finest sampling
        result = {}
        # todo (ktiwaery) fix this 1,-1,3 shape !!!!!! or else wont work with batchsize >1 
        # print("c_rgb.shape", c_rgb.shape, f_distance.shape)
        result['rgb_coarse'] = c_rgb.reshape(-1, 3)
        result['rgb_fine'] = f_rgb.reshape(-1, 3)
        result['depth_coarse'] = c_distance.reshape(-1, 1)
        result['depth_fine'] = f_distance.reshape(-1, 1)
        ret = torch.cat([result['rgb_fine'], result['rgb_coarse'], result['depth_coarse'], result['depth_fine']], -1)
        ret2 = torch.cat([f_weights, f_acc.reshape(-1, 1), f_t_samples], -1)
        return ret, ret2

    def render_rayschunk(rays_o: torch.Tensor, rays_d: torch.Tensor):
        # rays_o: [(B), N_rays, 3]
        # rays_d: [(B), N_rays, 3]
        prefix_batch = [B] if batched else []
        N_rays = rays_o.shape[-2]
        
        nears = near * torch.ones([*prefix_batch, N_rays, 1]).to(device)
        fars = far * torch.ones([*prefix_batch, N_rays, 1]).to(device)
        ret1, ret2 = batchify_query(_batched_forward, rays_o, rays_d, nears, fars)
        twob_ret = {}
        twob_ret['rgb_fine'] = ret1[:,:,0:3]
        twob_ret['rgb_coarse'] = ret1[:,:,3:6]
        twob_ret['depth_coarse'] = ret1[:,:,6:7]
        twob_ret['depth_fine'] = ret1[:,:,7:8]
        twob_ret['f_weights'] = ret2[:,:, 0:32]
        twob_ret['f_acc'] = ret2[:,:, 32]
        twob_ret['f_t_samples'] = ret2[:,:, 33:]
        return twob_ret

    ret = {}
    for i in tqdm(range(0, rays_o.shape[DIM_BATCHIFY], rayschunk), disable=not show_progress):
        ret_i = render_rayschunk(
            rays_o[:, i:i+rayschunk] if batched else rays_o[i:i+rayschunk],
            rays_d[:, i:i+rayschunk] if batched else rays_d[i:i+rayschunk],
        )
        for k, v in ret_i.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)
    for k, v in ret.items():
        # print("k:", k, len(v))
        ret[k] = torch.cat(v, DIM_BATCHIFY)
    
    # # NOTE: this is for debugging, which maintains computation graph. But not suitable for validation
    # ret = render_rayschunk(rays_o, rays_d)
    return ret['rgb_fine'], ret['depth_fine'], ret

class SingleRenderer(nn.Module):
    def __init__(self, model: VolSDF):
        super().__init__()
        self.model = model

    def forward(self, rays_o, rays_d, **kwargs):
        return volume_render(rays_o, rays_d, self.model, **kwargs)

class TwoBounceSingleRenderer(nn.Module):
    def __init__(self, model: VolSDF):
        super().__init__()
        self.model = model

    def forward(self, rays_o, rays_d, near, far, **kwargs):
        return volume_render_two_bounce(self.model, rays_o, rays_d, near, far, **kwargs)

class Trainer(nn.Module):
    def __init__(self, model: VolSDF, device_ids=[0], batched=True):
        super().__init__()
        self.model = model
        self.renderer = SingleRenderer(model)
        self.twob_renderer = TwoBounceSingleRenderer(model.mip_nerf)
        if len(device_ids) > 1:
            self.renderer = nn.DataParallel(self.renderer, device_ids=device_ids, dim=1 if batched else 0)
            self.twob_renderer = nn.DataParallel(self.twob_renderer, device_ids=device_ids, dim=1 if batched else 0)
        self.device = device_ids[0]
    
    def forward(self, 
             args,
             indices,
             model_input,
             ground_truth,
             render_kwargs_train: dict,
             it: int, 
             rays=None, 
             radii=None):
        device = self.device
        if rays is None: 
            intrinsics = model_input["intrinsics"].to(device)
            c2w = model_input['c2w'].to(device)
            H = render_kwargs_train['H']
            W = render_kwargs_train['W']
            if args.model.use_env_mlp == 'virtual_cone_caustic_approx_5drf_mip_nerf':
                rays_o, rays_d, select_inds = rend_util.get_rays(c2w, intrinsics, H, W, N_rays=args.data.N_rays, compute_radii=True)
                rays_radii = rays_d[..., 3]
                render_kwargs_train['rays_radii'] = rays_radii
                rays_d = rays_d[..., 0:3]
            else:
                rays_o, rays_d, select_inds = rend_util.get_rays(c2w, intrinsics, H, W, N_rays=args.data.N_rays, compute_radii=False)
        else:
            rays_o, rays_d, select_inds = rays
        # [B, N_rays, 3]
        if "mask_target_object" in args.training and args.training.mask_target_object:
            mask_target = torch.gather(model_input["object_mask"].to(device), 1, select_inds)
            if "mask_target_horizon" in args.training and args.training.mask_target_horizon:
                mask_target_horizon = torch.gather(model_input["horizon_mask"].to(device), 1, select_inds)
                mask_target = torch.logical_or(mask_target,mask_target_horizon)
        else:
            mask_target = None

        target_rgb = torch.gather(ground_truth['rgb'].to(device), 1, torch.stack(3*[select_inds],-1))
        if mask_target is not None:
            target_rgb = target_rgb*mask_target[...,None]

        if args.model.polarized:
            if args.data.gt_type == 'normal':
                # [B, N_rays,3]
                target_normal = torch.gather(ground_truth['normal'].to(device), 1, torch.stack(3*[select_inds],-1))
                if args.model.only_diffuse:
                    # [B, N_rays,3, 3]
                    target_stokes = stokes_from_normal_rad(rays_o, rays_d, target_normal, 
                                                        target_rgb, train_mode=True)
                else:
                    target_specular = torch.gather(ground_truth['specular'].to(device), 1, torch.stack(3*[select_inds],-1))
                    target_stokes = stokes_from_normal_rad(rays_o, rays_d, target_normal, 
                                                        target_rgb, spec_rads=target_specular, 
                                                        train_mode=True)
            elif args.data.gt_type == 'stokes':
                target_s0 = torch.gather(ground_truth['s0'].to(device), 1, torch.stack(3*[select_inds],-1))
                target_s1 = torch.gather(ground_truth['s1'].to(device), 1, torch.stack(3*[select_inds],-1))
                target_s2 = torch.gather(ground_truth['s2'].to(device), 1, torch.stack(3*[select_inds],-1))
                target_stokes = torch.stack([target_s0, target_s1, target_s2], -1)
            else:
                raise Exception(f'Invalid data gt_type {args.data.gt_type}. Options: stokes, normal')
        # [B, N_rays]
        if mask_target is not None and args.model.polarized:
            target_stokes = target_stokes*mask_target[...,None,None]

        if "use_mask" in args.model and args.model.use_mask:
            mask_ignore = torch.gather(model_input["object_mask"].to(device), 1, select_inds)
            if "mask_target_horizon" in args.training and args.training.mask_target_horizon:
                mask_target_horizon = torch.gather(model_input["horizon_mask"].to(device), 1, select_inds)
                mask_ignore = torch.logical_or(mask_ignore,mask_target_horizon)
        else:
            mask_ignore = None 
        if "mask_specular" in args.training and args.training.mask_specular:
            mask_spec = torch.gather(model_input["object_mask"].to(device), 1, select_inds)
            if "mask_target_horizon" in args.training and args.training.mask_target_horizon:
                mask_target_horizon = torch.gather(model_input["horizon_mask"].to(device), 1, select_inds)
                mask_spec = torch.logical_or(mask_spec,mask_target_horizon)
        else:
            mask_spec = None

        # For diffuse and specular, disable specular rendering 
        if not args.model.only_diffuse:
            if (not args.two_bounce.training.include_polarization) and it < args.training.num_no_s1_s2:
                render_kwargs_train['only_diffuse'] = True
            else: 
                render_kwargs_train['only_diffuse'] = False

        # if use refl_dirs as dirs 
        if args.two_bounce.training.use_refl_as_vdirs:
            if it < args.two_bounce.training.start_refl_as_vdirs_at:
                # use virtual_dirs first to get good specular 
                render_kwargs_train['use_refl_as_vdirs'] = False
            else: 
                # Start using reflec_dirs after flag
                render_kwargs_train['use_refl_as_vdirs'] = True

        rgb, depth_v, extras = self.renderer(rays_o, rays_d, detailed_output=True, **render_kwargs_train)

        # [B, N_rays, N_pts, 3]
        nablas: torch.Tensor = extras['implicit_nablas']
        # extras['depth_v'] = depth_v
        extras['rays_o'] = rays_o
        extras['rays_d'] = rays_d
        
        # [B, N_rays, ]
        #---------- OPTION1: just flatten and use all nablas
        # nablas = nablas.flatten(-3, -2)
        
        #---------- OPTION2: using only one point each ray: this may be what the paper suggests.
        # @ VolSDF section 3.5, "combine a SINGLE random uniform space point and a SINGLE point from \mathcal{S} for each pixel"
        _, _ind = extras['visibility_weights'][..., :nablas.shape[-2]].max(dim=-1)
        nablas = torch.gather(nablas, dim=-2, index=_ind[..., None, None].repeat([*(len(nablas.shape)-1)*[1], 3]))
        
        eik_bounding_box = args.model.obj_bounding_radius
        eikonal_points = torch.empty_like(nablas).uniform_(-eik_bounding_box, eik_bounding_box).to(device)
        _, nablas_eik, _ = self.model.implicit_surface.forward_with_nablas(eikonal_points)
        nablas = torch.cat([nablas, nablas_eik], dim=-2)

        # [B, N_rays, N_pts]
        nablas_norm = torch.norm(nablas, dim=-1)

        losses = OrderedDict()

        if args.training.loss_space == 'linear':
            to_loss_space = lambda x : x
        elif args.training.loss_space == 'srgb':
            to_loss_space = lambda x : linear_rgb_to_srgb(x)
        else:
            raise Exception(f'Invalid loss space {args.training.loss_space}') 
        
        if args.training.loss_type == 'l1':
            loss_fn = lambda x,y : F.l1_loss(to_loss_space(x), to_loss_space(y), reduction='none')
        elif args.training.loss_type == 'mse':
            loss_fn = lambda x,y : F.mse_loss(to_loss_space(x), to_loss_space(y), reduction='none')
        elif args.training.loss_type == 'weighted_mse':
            #Same as rawNeRF https://arxiv.org/pdf/2111.13679.pdf
            loss_fn = lambda x,y : F.mse_loss(to_loss_space(x), to_loss_space(y), reduction='none')\
                                    /(to_loss_space(x.detach())+1e-3)**2

        if 's1_coarse' in extras.keys() or 'spec_s1_coarse' in extras.keys():
            # print("flag_use_specular set to True")
            extras['flag_use_coarse_specular'] = True 
        else:
            # print("flag_use_specular set to False")
            extras['flag_use_coarse_specular'] = False

        if args.model.polarized:
            if (not args.two_bounce.training.include_polarization) \
                        and (args.model.only_diffuse or (it < args.training.num_no_s1_s2)): 
                s0 = extras['s0']
                s1 = extras['s1']
                s2 = extras['s2']
                if extras['flag_use_coarse_specular']:
                    s0_coarse = extras['s0_coarse']
                    s1_coarse = extras['s1_coarse']
                    s2_coarse = extras['s2_coarse']

            else:
                if mask_spec is not None :
                    s0 = extras['diff_s0'] + mask_spec[...,None]*extras['spec_s0']
                    s1 = extras['diff_s1'] + mask_spec[...,None]*extras['spec_s1']
                    s2 = extras['diff_s2'] + mask_spec[...,None]*extras['spec_s2']
                    if extras['flag_use_coarse_specular']:
                        s0_coarse = extras['diff_s0'] + mask_spec[...,None]*extras['spec_s0_coarse']
                        s1_coarse = extras['diff_s1'] + mask_spec[...,None]*extras['spec_s1_coarse']
                        s2_coarse = extras['diff_s2'] + mask_spec[...,None]*extras['spec_s2_coarse']
                else:
                    # print("SHOULD BE HERE! mask spec is none")
                    s0 = extras['diff_s0'] + extras['spec_s0']
                    s1 = extras['diff_s1'] + extras['spec_s1']
                    s2 = extras['diff_s2'] + extras['spec_s2']
                    if extras['flag_use_coarse_specular']:
                        s0_coarse = extras['diff_s0'] + extras['spec_s0_coarse']
                        s1_coarse = extras['diff_s1'] + extras['spec_s1_coarse']
                        s2_coarse = extras['diff_s2'] + extras['spec_s2_coarse']

            losses['loss_img'] = loss_fn(s0, target_stokes[...,0])

            if extras['flag_use_coarse_specular']:
                losses['loss_img_coarse'] = loss_fn(s0_coarse, target_stokes[...,0])

            if it > args.training.num_no_s1_s2:
                w_s1_s2 = args.training.w_s1_s2
            else:
                w_s1_s2 = 0.

            # compute loss s1_s2
            losses['loss_s1_s2'] = w_s1_s2*loss_fn(s1, target_stokes[...,1])+\
                                w_s1_s2*loss_fn(s2, target_stokes[...,2])                                
            if extras['flag_use_coarse_specular']:
                losses['loss_s1_s2_coarse'] = w_s1_s2*loss_fn(s1_coarse.reshape(s1.shape), target_stokes[...,1])+\
                                            w_s1_s2*loss_fn(s2_coarse.reshape(s1.shape), target_stokes[...,2])

            # losses['loss_img'] = loss_fn(s0, target_stokes[...,0])
            if 'fres_out' in extras.keys():
                losses['loss_fres'] = args.training.w_fres*extras['fres_diff']
        else:
            # Compute Loss only on RGB! 
            if args.model.only_diffuse:
                # this computes loss only on the diffuse component. 
                losses['loss_img'] = loss_fn(rgb, target_rgb)
            else:
                if args.data.gt_type == 'normal':
                    target_specular = torch.gather(ground_truth['specular'].to(device), 1, torch.stack(3*[select_inds],-1))
                    mixed_target = target_rgb + target_specular
                elif args.data.gt_type == 'stokes':
                    mixed_target = target_rgb

                if it < args.training.num_no_s1_s2:
                    # this will compute loss on the diffuse+specular component, but only use the diffuse 
                    mixed_ours = rgb
                else:
                    if mask_spec is not None:
                        mixed_ours = mask_spec[..., None]*extras['spec_map'] + rgb
                        if extras['flag_use_coarse_specular']:
                            mixed_ours_coarse = mask_spec[..., None]*extras['spec_map_coarse'] + rgb
                    else:
                        mixed_ours = extras['spec_map'] + rgb
                        if extras['flag_use_coarse_specular']:
                            mixed_ours_coarse = extras['spec_map_coarse'] + rgb

                losses['loss_img'] = loss_fn(mixed_ours, mixed_target)
                if extras['flag_use_coarse_specular']:
                    losses['loss_img_coarse'] = loss_fn(mixed_ours_coarse, mixed_target)
                #################

        losses['loss_eikonal'] = args.training.w_eikonal * F.mse_loss(nablas_norm, nablas_norm.new_ones(nablas_norm.shape), reduction='mean')
        # print("losses['loss_eikonal']", losses['loss_eikonal'])

        if args.training.normal_reg_loss:
            v = -1. * extras['rays_d']
            n_dot_v = (extras['implicit_nablas'][..., :-1, :] * v[..., None, :]).sum(axis=-1)
            losses['loss_normal_orientation'] = torch.mean((extras['visibility_weights'] * torch.minimum(torch.tensor(0.0), n_dot_v)**2).sum(axis=-1))
        
        """
        try:
            if args.training.dist_loss:
                if 'f_weights' in extras: 
                    losses['f_distloss'] = args.training.w_dist_loss * distloss(extras["f_weights"], extras["f_t_samples"])
                    print("using dist loss", losses['f_distloss'])
        except KeyError: 
            pass
        """

        if mask_ignore is not None:
            losses['loss_img'] = (losses['loss_img'] * mask_ignore[..., None].float()).sum() / (mask_ignore.sum() + 1e-10)
            if extras['flag_use_coarse_specular']:
                losses['loss_img_coarse'] = args.two_bounce.loss.coarse_loss_mult * \
                            (losses['loss_img_coarse'] * mask_ignore[..., None].float()).sum() / (mask_ignore.sum() + 1e-10)
            if args.model.polarized:
                losses['loss_s1_s2'] = (losses['loss_s1_s2'] * mask_ignore[..., None].float()).sum() / (mask_ignore.sum() + 1e-10)
                if extras['flag_use_coarse_specular']:
                    losses['loss_s1_s2_coarse'] = args.two_bounce.loss.coarse_loss_mult * \
                                (losses['loss_s1_s2_coarse'] * mask_ignore[..., None].float()).sum() / (mask_ignore.sum() + 1e-10)
                # if 'loss_fres' in losses.keys():
                #     losses['loss_fres'] = (losses['loss_fres'] * mask_ignore[..., None].float()).sum() / (mask_ignore.sum() + 1e-10)

        else:
            losses['loss_img'] = losses['loss_img'].mean()
            if extras['flag_use_coarse_specular']:
                losses['loss_img_coarse'] = args.two_bounce.loss.coarse_loss_mult * losses['loss_img_coarse'].mean()
            if args.model.polarized:
                losses['loss_s1_s2'] = losses['loss_s1_s2'].mean()
                if extras['flag_use_coarse_specular']:
                    losses['loss_s1_s2_coarse'] = args.two_bounce.loss.coarse_loss_mult * losses['loss_s1_s2_coarse'].mean()
                # if 'loss_fres' in losses.keys():
                #     losses['loss_fres'] = losses['loss_fres'].mean()


        # Regularize specular component for sparsity
        if not args.model.only_diffuse:
            if it > args.training.num_no_s1_s2:
                losses['loss_spec_reg'] = args.training.w_spec_reg*extras['spec_map'].abs().mean()
                if extras['flag_use_coarse_specular']:
                    losses['loss_spec_reg_coarse'] = args.training.w_spec_reg*extras['spec_map_coarse'].abs().mean()

        # From neus.py
        if args.training.w_mask > 0. :
            # mask_volume = extras['mask_volume']
            mask_volume = extras['mask_map']
            # mask_volume = torch.clamp(mask_volume, 0, 1)
            # NOTE: when predicted mask is close to 1 but GT is 0, exploding gradient.
            mask_volume = torch.clamp(mask_volume, 1e-10, 1-1e-10)
            # mask_volume = torch.clamp(mask_volume, 1e-3, 1-1e-3)
            extras['mask_volume'] = mask_volume
            target_mask = torch.gather(model_input["object_mask"].to(device), 1, select_inds)
            if "mask_target_horizon" in args.training and args.training.mask_target_horizon:
                mask_target_horizon = torch.gather(model_input["horizon_mask"].to(device), 1, select_inds)
                target_mask = torch.logical_or(target_mask,mask_target_horizon)
            # BCE loss
            # target_mask = torch.clamp(target_mask, 0, 1)
            # print("mask_volume", mask_volume.shape, target_mask.shape)
            # print("mask_volume", mask_volume.max(), target_mask.max())
            # CHANGED 
            losses['loss_mask'] = args.training.w_mask * F.binary_cross_entropy(mask_volume, target_mask.float(), reduction='mean')
            # losses['loss_mask'] = args.training.w_mask * self.model.mask_loss(mask_volume, target_mask.float())
            
            # Only set 0 labels in BCE loss
            # losses['loss_mask'] = args.training.w_mask * ((target_mask.float()-1)*torch.log(1 - mask_volume)).sum()/(target_mask.float().sum()-1+1e-10)

        # Distortion Loss
        if args.two_bounce.mip_nerf.use_mip_nerf and args.two_bounce.loss.use_distloss \
                and 'dloss_fine' in list(extras.keys()) and 'dloss_coarse' in list(extras.keys()):
            losses['loss_img'] += extras['dloss_fine'].sum()
            if 'loss_img_coarse' in list(losses.keys()):
                losses['loss_img_coarse'] += extras['dloss_coarse'].sum()
            #print("Distortion loss - fine {}, coarse {}".format(extras['dloss_fine'].mean(), extras['dloss_coarse'].mean()))

        loss = 0
        for k, v in losses.items():
            # print("k {} -> {} -> {}".format(k, losses[k], torch.isnan(losses[k]).any()))
            if it <= args.training.mask_only_until: 
                if k == "loss_mask" or k == "loss_normal_orientation":
                    loss += losses[k]
            else:
                loss += losses[k]
            assert not torch.isnan(losses[k]).any()
            
        
        losses['total'] = loss
        
        extras['implicit_nablas_norm'] = nablas_norm

        alpha, beta = self.model.forward_ab()
        alpha = alpha.data
        beta = beta.data
        extras['scalars'] = {'beta': beta, 'alpha': alpha}
        extras['select_inds'] = select_inds

        return OrderedDict(
            [('losses', losses),
             ('extras', extras)])
        

    def val(self, logger: Logger, ret, to_img_fn, it, render_kwargs_test):
        #----------- plot beta heat map
        beta_heat_map = to_img_fn(ret['beta_map']).permute(0, 2, 3, 1).data.cpu().numpy()
        beta_heat_map = io_util.gallery(beta_heat_map, int(np.sqrt(beta_heat_map.shape[0])))
        _, beta = self.model.forward_ab()
        beta = beta.data.cpu().numpy().item()
        # beta_min = beta_heat_map.min()
        beta_max = beta_heat_map.max().item()
        if beta_max != beta:
            ticks = np.linspace(beta, beta_max, 10).tolist()
        else:
            ticks = [beta]
        tick_labels = ["{:.4f}".format(b) for b in ticks]
        tick_labels[0] = "beta={:.4f}".format(beta)
        
        fig = plt.figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax_im = ax.imshow(beta_heat_map, vmin=beta, vmax=beta_max)
        cbar = fig.colorbar(ax_im, ticks=ticks)
        cbar.ax.set_yticklabels(tick_labels)
        logger.add_figure(fig, 'val/beta_heat_map', it)
        
        #----------- plot iteration used for each ray
        max_iter = render_kwargs_test['max_upsample_steps']
        iter_usage_map = to_img_fn(ret['iter_usage'].unsqueeze(-1)).permute(0, 2, 3, 1).data.cpu().numpy()
        iter_usage_map = io_util.gallery(iter_usage_map, int(np.sqrt(iter_usage_map.shape[0])))
        iter_usage_map[iter_usage_map==-1] = max_iter+1
        
        fig = plt.figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax_im = ax.imshow(iter_usage_map, vmin=0, vmax=max_iter+1)
        ticks = list(range(max_iter+2))
        tick_labels = ["{:d}".format(b) for b in ticks]
        tick_labels[-1] = 'not converged'
        cbar = fig.colorbar(ax_im, ticks=ticks)
        cbar.ax.set_yticklabels(tick_labels)
        logger.add_figure(fig, 'val/upsample_iters', it)

def get_model(args):
    model_config = {
        'use_nerfplusplus': args.model.setdefault('outside_scene', 'builtin') == 'nerf++',
        'obj_bounding_radius': args.model.obj_bounding_radius,
        'W_geo_feat': args.model.setdefault('W_geometry_feature', 256),
        'speed_factor': args.training.setdefault('speed_factor', 1.0),
        'beta_init': args.training.setdefault('beta_init', 0.1),
        'sh_embed': args.model.setdefault('sh_embed',False),
        'polarized': args.model.setdefault('polarized',False),
        'pol_render_type': args.model.setdefault('pol_render_type','stokes_accum'),
        'only_diffuse': args.model.setdefault('only_diffuse',False),
        'env_mlp_type': args.model.setdefault('env_mlp_type', 'no_fres'),
        'fine_spec': args.model.setdefault('fine_spec', False),
        'use_env_mlp': args.model.setdefault('use_env_mlp', "no_envmap_MLP"),
        'disable_fres': args.model.setdefault('disable_fres', False),
    }
    
    surface_cfg = {
        'use_siren': args.model.surface.setdefault('use_siren', args.model.setdefault('use_siren', False)),
        'embed_multires': args.model.surface.setdefault('embed_multires', 6),
        'radius_init':  args.model.surface.setdefault('radius_init', 1.0),
        'geometric_init': args.model.surface.setdefault('geometric_init', True),
        'D': args.model.surface.setdefault('D', 8),
        'W': args.model.surface.setdefault('W', 256),
        'skips': args.model.surface.setdefault('skips', [4]),
    }
        
    radiance_cfg = {
        'use_siren': args.model.radiance.setdefault('use_siren', args.model.setdefault('use_siren', False)),
        'embed_multires': args.model.radiance.setdefault('embed_multires', -1),
        'embed_multires_view': args.model.radiance.setdefault('embed_multires_view', -1),
        'use_view_dirs': args.model.radiance.setdefault('use_view_dirs', True),
        'D': args.model.radiance.setdefault('D', 4),
        'W': args.model.radiance.setdefault('W', 256),
        'skips': args.model.radiance.setdefault('skips', []),
    }
    
    model_config['surface_cfg'] = surface_cfg
    model_config['radiance_cfg'] = radiance_cfg

    two_bounce_cfg = {}
    if args.two_bounce.mip_nerf.use_mip_nerf:
        two_bounce_cfg['mip_nerf_cfg'] = args.two_bounce.mip_nerf
    else:
        two_bounce_cfg['nerf_cfg'] = args.two_bounce.mip_nerf

    ## add detach from training: 
    try:
        two_bounce_cfg['remove_diffuse_component'] = args.training.remove_diffuse_component
        print("Diffuse Component set to {}!".format(args.training.remove_diffuse_component))
    except KeyError: 
        two_bounce_cfg['remove_diffuse_component'] = False
        print("Not Removing Diffuse Component!")

    two_bounce_cfg['detach_radii'] = args.two_bounce.training.setdefault('detach_radii', False)
    two_bounce_cfg['detach_nablas'] = args.two_bounce.training.setdefault('detach_nablas', False)
    two_bounce_cfg['detach_x'] = args.two_bounce.training.setdefault('detach_x', False)
    
    try: 
        use_refl_as_vdirs = args.two_bounce.training.use_refl_as_vdirs
        print("---- NOTE: Using Reflected Rays as Virtual Ray Dirs: {} ----".format(use_refl_as_vdirs))
    except KeyError:
        use_refl_as_vdirs = False
        print("---- NOTE: NOT USING REFL RAYS as VIRTUAL RAY DIRS ----")


    try: 
        twob_near = args.two_bounce.near
        twob_far = args.two_bounce.far
        print("Setting Two Bounce Bounds to ({},{})".format(twob_near, twob_far))
    except KeyError:
        import warnings
        twob_near = 0 #args.two_bounce.near
        twob_far = 6 #args.two_bounce.far
        warnings.warn("Defaulting two bounce near and far bounds to 0 and 6! Please set this explicitly in the config under args.two_bounce.near/far", DeprecationWarning)
        raise ValueError("Defaulting two bounce near and far bounds to 0 and 6! Please set this explicitly in the config under args.two_bounce.near/far")

    print("NOTE: Detaching Nablas: {}, Radii: {} and x: {}".format(
        two_bounce_cfg['detach_nablas'], 
        two_bounce_cfg['detach_radii'], 
        two_bounce_cfg['detach_x']
    ))

    model_config['two_bounce_cfg'] = two_bounce_cfg

    model = VolSDF(**model_config)
    
    ## render_kwargs
    render_kwargs_train = {
        'near': args.data.near,
        'far': args.data.far,
        'twob_near': twob_near,
        'twob_far': twob_far,
        'batched': True,
        'perturb': args.model.setdefault('perturb', True),   # config whether do stratified sampling
        'white_bkgd': args.model.setdefault('white_bkgd', False),
        'max_upsample_steps': args.model.setdefault('max_upsample_iter', 5),
        'use_nerfplusplus': args.model.setdefault('outside_scene', 'builtin') == 'nerf++',
        'obj_bounding_radius': args.model.obj_bounding_radius,
        'polarized': args.model.setdefault('polarized',False),
        'sh_embed': args.model.setdefault('sh_embed',False),
        'pol_render_type': args.model.setdefault('pol_render_type','stokes_accum'),
        'only_diffuse': args.model.setdefault('only_diffuse',False),
        'N_samples': args.model.setdefault('N_samples',128),
        'use_env_mlp': args.model.setdefault('use_env_mlp','no_envmap_MLP'),
        'use_depth_for_two_bounce': args.two_bounce.mip_nerf.setdefault('use_depth_for_two_bounce', False), 
        'nan_to_num_enabled': args.two_bounce.loss.setdefault('nan_to_num_enabled', True), 
        'use_refl_as_vdirs': use_refl_as_vdirs
    }

    render_kwargs_test = copy.deepcopy(render_kwargs_train)
    render_kwargs_test['rayschunk'] = args.data.val_rayschunk
    render_kwargs_test['perturb'] = False
    
    trainer = Trainer(model, args.device_ids, batched=render_kwargs_train['batched'])
    
    return model, trainer, render_kwargs_train, render_kwargs_test, trainer.renderer
