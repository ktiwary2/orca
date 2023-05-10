from typing import Callable
import dotdict
from einops import repeat
from tqdm import tqdm
from utils.two_bounce_utils import sample_along_rays, distloss, lossfun_distortion, get_reflected_ray, integrated_pos_enc, pos_enc, volumetric_rendering, resample_along_rays
from utils import io_util, train_util, rend_util
from src.utils import  linear_rgb_to_srgb

from collections import OrderedDict, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

perpare_specular = lambda x: torch.clip(linear_rgb_to_srgb(x), 0, 1)

def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)

class MLP(torch.nn.Module):
    """
    A simple MLP.
    """

    def __init__(self, net_depth: int, net_width: int, net_depth_condition: int, net_width_condition: int,
                 skip_index: int, num_rgb_channels: int, num_density_channels: int, activation: str,
                 xyz_dim: int, view_dim: int):
        """
          net_depth: The depth of the first part of MLP.
          net_width: The width of the first part of MLP.
          net_depth_condition: The depth of the second part of MLP.
          net_width_condition: The width of the second part of MLP.
          activation: The activation function.
          skip_index: Add a skip connection to the output of every N layers.
          num_rgb_channels: The number of RGB channels.
          num_density_channels: The number of density channels.
        """
        super(MLP, self).__init__()
        self.skip_index: int = skip_index  # Add a skip connection to the output of every N layers.
        layers = []
        for i in range(net_depth):
            if i == 0:
                dim_in = xyz_dim
                dim_out = net_width
            elif (i - 1) % skip_index == 0 and i > 1:
                dim_in = net_width + xyz_dim
                dim_out = net_width
            else:
                dim_in = net_width
                dim_out = net_width
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            if activation == 'relu':
                layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
            elif activation == 'softplus':
                layers.append(torch.nn.Sequential(linear, torch.nn.Softplus()))
            else:
                raise NotImplementedError("activation {} not found.".format(activation))
        self.layers = torch.nn.ModuleList(layers)
        del layers
        self.density_layer = torch.nn.Linear(net_width, num_density_channels)
        _xavier_init(self.density_layer)
        self.extra_layer = torch.nn.Linear(net_width, net_width)  # extra_layer is not the same as NeRF
        _xavier_init(self.extra_layer)
        layers = []
        for i in range(net_depth_condition):
            if i == 0:
                dim_in = net_width + view_dim
                dim_out = net_width_condition
            else:
                dim_in = net_width_condition
                dim_out = net_width_condition
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            if activation == 'relu':
                layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
            elif activation == 'softplus':
                layers.append(torch.nn.Sequential(linear, torch.nn.Softplus()))
            else:
                raise NotImplementedError("activation {} not found.".format(activation))
        self.view_layers = torch.nn.Sequential(*layers)
        del layers
        self.color_layer = torch.nn.Linear(net_width_condition, num_rgb_channels)

    def forward(self, x, view_direction=None):
        """Evaluate the MLP.

        Args:
            x: torch.Tensor(float32), [batch, num_samples, feature], points.
            view_direction: torch.Tensor(float32), [batch, feature], if not None, this
            variable will be part of the input to the second part of the MLP
            concatenated with the output vector of the first part of the MLP. If
            None, only the first part of the MLP will be used with input x. In the
            original paper, this variable is the view direction.

        Returns:
            raw_rgb: torch.Tensor(float32), with a shape of
                [batch, num_samples, num_rgb_channels].
            raw_density: torch.Tensor(float32), with a shape of
                [batch, num_samples, num_density_channels].
        """
        num_samples = x.shape[1]
        inputs = x  # [B, N, 2*3*L]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i % self.skip_index == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        raw_density = self.density_layer(x)
        if view_direction is not None:
            # Output of the first part of MLP.
            bottleneck = self.extra_layer(x)
            # Broadcast condition from [batch, feature] to
            # [batch, num_samples, feature] since all the samples along the same ray
            # have the same viewdir.
            # view_direction: [B, 2*3*L] -> [B, N, 2*3*L]
            view_direction = repeat(view_direction, 'batch feature -> batch sample feature', sample=num_samples)
            x = torch.cat([bottleneck, view_direction], dim=-1)
            # Here use 1 extra layer to align with the original nerf model.
            x = self.view_layers(x)
        raw_rgb = self.color_layer(x)
        return raw_rgb, raw_density

class MipNerf(torch.nn.Module):
    """Nerf NN Model with both coarse and fine MLPs."""

    def __init__(self, args):
        super(MipNerf, self).__init__()
        self.args = args
        self.num_levels = self.args.num_levels  # The number of sampling levels.
        self.num_samples = self.args.num_samples  # The number of samples per level.
        print("Using {} samples for mipnerf".format(self.num_samples))
        self.disparity = self.args.disparity  # If True, sample linearly in disparity, not in depth.
        self.ray_shape = self.args.ray_shape  # The shape of cast rays ('cone' or 'cylinder').
        self.disable_integration = self.args.disable_integration  # If True, use PE instead of IPE.
        self.min_deg_point = self.args.min_deg_point  # Min degree of positional encoding for 3D points.
        self.max_deg_point = self.args.max_deg_point  # Max degree of positional encoding for 3D points.
        self.use_viewdirs = self.args.use_viewdirs  # If True, use view directions as a condition.
        self.deg_view = self.args.deg_view  # Degree of positional encoding for viewdirs.
        self.density_noise = self.args.density_noise  # Standard deviation of noise added to raw density.
        self.density_bias = self.args.density_bias  # The shift added to raw densities pre-activation.
        self.resample_padding = self.args.resample_padding  # Dirichlet/alpha "padding" on the histogram.
        self.stop_resample_grad = self.args.stop_resample_grad  # If True, don't backprop across levels')
        self.mlp_xyz_dim = (self.max_deg_point - self.min_deg_point) * 3 * 2
        try: 
            if self.args.add_geom_features:
                self.mlp_xyz_dim += self.args.add_geom_features
                print("Adding geometry features to mipnerf")
            else:
                print("Not adding geometry features to mipnerf")
        except KeyError:
            print("Not adding geometry features to mipnerf")
        self.mlp_view_dim = self.deg_view * 3 * 2
        self.mlp_view_dim = self.mlp_view_dim + 3 if self.args.append_identity else self.mlp_view_dim
        try: 
            if self.args.add_ide_features:
                self.mlp_view_dim += self.args.add_ide_features
                print("Adding IDE features to mipnerf")
            else:
                print("Not adding IDE features to mipnerf")
        except KeyError:
            print("Not adding IDE features to mipnerf")

        self.mlp = MLP(self.args.mlp.net_depth, 
                       self.args.mlp.net_width, 
                       self.args.mlp.net_depth_condition, 
                       self.args.mlp.net_width_condition,
                       self.args.mlp.skip_index, 
                       self.args.mlp.num_rgb_channels, 
                       self.args.mlp.num_density_channels, 
                       self.args.mlp.net_activation,
                       self.mlp_xyz_dim, self.mlp_view_dim)
        if self.args.rgb_activation == 'swish':  # The RGB activation.
            # self.rgb_activation = torch.nn.Sigmoid()
            self.rgb_activation = torch.nn.SiLU()
            # self.rgb_activation = torch.nn.SiLU()
        elif self.args.rgb_activation == 'softplus':  # The RGB activation.
            self.rgb_activation = torch.nn.Softplus()
        else:
            raise NotImplementedError('use swish or softplus')

        self.rgb_padding = self.args.rgb_padding
        if self.args.density_activation == 'softplus':  # Density activation.
            self.density_activation = torch.nn.Softplus()
        else:
            raise NotImplementedError

    def forward(self, rays: namedtuple, randomized: bool, white_bkgd: bool):
        """The mip-NeRF Model.
        Args:
            rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
            randomized: bool, use randomized stratified sampling.
            white_bkgd: bool, if True, use white as the background (black o.w.).
        Returns:
            ret: list, [*(rgb, distance, acc)]
        """

        ret = []
        t_samples, weights = None, None
        for i_level in range(self.num_levels):
            # key, rng = random.split(rng)
            if i_level == 0:
                # Stratified sampling along rays
                t_samples, means_covs = sample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    self.num_samples,
                    rays.near,
                    rays.far,
                    randomized,
                    self.disparity,
                    self.ray_shape,
                )
            else:
                t_samples, means_covs = resample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    t_samples,
                    weights,
                    randomized,
                    self.ray_shape,
                    self.stop_resample_grad,
                    resample_padding=self.resample_padding,
                )
            if self.disable_integration:
                means_covs = (means_covs[0], torch.zeros_like(means_covs[1]))
            samples_enc = integrated_pos_enc(
                means_covs,
                self.min_deg_point,
                self.max_deg_point,
            )  # samples_enc: [B, N, 2*3*L]  L:(max_deg_point - min_deg_point)

            # Point attribute predictions
            if self.use_viewdirs:
                viewdirs_enc = pos_enc(
                    rays.viewdirs,
                    min_deg=0,
                    max_deg=self.deg_view,
                    append_identity=True,
                )
                # print("mipnerf input shape: ", samples_enc.shape, viewdirs_enc.shape)
                raw_rgb, raw_density = self.mlp(samples_enc, viewdirs_enc)
            else:
                raw_rgb, raw_density = self.mlp(samples_enc)

            # Add noise to regularize the density predictions if needed.
            if randomized and (self.density_noise > 0):
                raw_density += self.density_noise * torch.randn(raw_density.shape, dtype=raw_density.dtype)

            # Volumetric rendering.
            rgb = self.rgb_activation(raw_rgb)  # [B, N, 3]
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            density = self.density_activation(raw_density + self.density_bias)  # [B, N, 1]
            comp_rgb, distance, acc, weights = volumetric_rendering(
                rgb,
                density,
                t_samples,
                rays.directions,
                white_bkgd=white_bkgd,
            )
            ret.append((comp_rgb, distance, acc, weights, t_samples))

        return ret

class MipNerfTrainer(nn.Module):
    def __init__(self, model: MipNerf, train_type: str = '5dreflected_only', device_ids=[0], batched=True):
        """
        train_type: choose from ["5dreflected_only", "5dspec_lobe"]
        """
        super().__init__()
        self.model = model
        if len(device_ids) > 1: 
            print('calling data parallel on model, with batched: {}'.format(batched))
            self.model = nn.DataParallel(self.model, device_ids=device_ids, dim=0 if batched else 0)

        # if len(device_ids) > 1:
        #     self.renderer = nn.DataParallel(self.renderer, device_ids=device_ids, dim=1 if batched else 0)
        # self.device = device_ids[0]
        self.device_ids = device_ids
        self.train_type = train_type
        self.all_types= ["5dreflected_only", "5dspec_lobe"]
        self.to_loss_space = None
        self.loss_fn = None
   
    def forward(self, 
             args,
             render_kwargs: dict,
             near, 
             far,
             use_input_rays_as_query_rays=False,
             compute_loss=True,
             randomized=True,
            # render algorithm config
             rayschunk = 2048,
             netchunk = 1048576,
             white_bkgd = False,
            # render function config
            detailed_output = True,
            show_progress = False,
             ):

        ###################
        def batchify_rays(rays_o: torch.Tensor, rays_d: torch.Tensor, 
                          gt_radiance: torch.Tensor, loss_fn: Callable, 
                          batched=True,
                          ):
            # rays_o: [(B), N_rays, 3]
            # rays_d: [(B), N_rays, 3]
            rays = dotdict.dotdict()
            rays.origins = rays_o # (B, N, 3)
            rays.directions = rays_d # (B, N, 3)
            rays.viewdirs = rays_d # volsdf everything is normalized so should be the same 
            # print("batchify_rays:",rays_o.shape, rays_d.shape)


            # TODO: (ktiwary) not sure this is correct!
            # Distance from each unit-norm direction vector to its x-axis neighbor.
            # radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]
            dx = torch.sqrt(torch.sum((rays_d[:, :-1, :] - rays_d[:, 1:, :]) ** 2, -1))
            dx = torch.concat([dx, dx[:, -2:-1]], 1)

            if rays_d.shape[1] != dx.shape[1]: # happes when ray_d.shape == [1,2,1]
                dx = torch.cat([dx,dx], 1)

            # # Cut the distance in half, and then round it out so that it's
            # # halfway between inscribed by / circumscribed about the pixel.
            rays.radii = dx[...,None] * 2 / torch.sqrt(torch.tensor(12))
            # print("rays.radii.max()", rays.radii.max())
            if batched:
                B = rays_o.shape[0]
                N_rays = rays_o.shape[1]
                shape = [B*N_rays, 1]
            else:
                N_rays = rays_o.shape[0]
                shape = [N_rays, 1]
            rays.near = near * torch.ones(shape).to(device)
            rays.far = far * torch.ones(shape).to(device)
            # print("rays.directions.shape {} and rays.origins.shape {}, rays.radii.shape {} --> {}".format(rays.directions.shape, 
            #                                 rays.origins.shape, rays.radii.shape, shape))
 
            # reshape all rays 
            rays.origins = rays.origins.reshape(-1, 3)
            rays.directions = rays.directions.reshape(-1, 3)
            rays.viewdirs = rays.viewdirs.reshape(-1, 3)
            rays.radii = rays.radii.reshape(-1, 1)
            rays.near = rays.near.reshape(-1, 1)
            rays.far = rays.far.reshape(-1, 1)

            # print("rays.origins.shape: {}, gt.shape: {}, rgb.shape: {},  distance.shape: {}".format(
            #                         rays.origins.shape, gt_radiance.shape, rays.directions.shape, rays.near.shape))
                                    
            ret = self.model.forward(rays=rays, randomized=randomized, white_bkgd=white_bkgd)
            # print('ret', len(ret), len(ret[0]), len(ret[1]))
            # (c_rgb, _, _, _, _), (f_rgb, distance, _, _, _) 
            c_rgb, c_distance, _, _, _ = ret[0] # first pass is coarse sampling 
            f_rgb, f_distance, f_acc, f_weights, f_t_samples = ret[-1] # the last one is the finest sampling

            # 3) compute losses
            ret_i = {}
            # print("c_rgb.shape", c_rgb.shape, f_distance.shape, f_acc.shape, f_weights.shape, f_t_samples.shape)
            ret_i['c_rgb'] = c_rgb.reshape(1, -1, 3)
            ret_i['f_rgb'] = f_rgb.reshape(1, -1, 3)
            ret_i['f_distance'] = f_distance.reshape(1, -1, 1)
            if compute_loss:
                losses = []
                distlosses = []
                for _, (rgb, _, _, weights, t_samples) in enumerate(ret):
                    # L2 loss changed according to 
                    #  https://github.com/google-research/multinerf/blob/47fad9688748b3cc962990c19898aff78b45968e/internal/train_utils.py#L105 
                    if args.two_bounce.loss.bound_gt:
                        assert perpare_specular(gt_radiance).max() <= 1.0
                        gt_radiance = perpare_specular(gt_radiance)
                        # raise ValueError("You shouldn't do this...")
                    _loss = loss_fn(rgb, gt_radiance.reshape(rgb.shape))
                    if args.two_bounce.loss.normalize_loss: 
                    # if False: 
                        lossmult = torch.ones(rgb.shape).to(rgb.device) # used to weight each ray
                        denom = lossmult.sum() 
                        loss = (lossmult * _loss).sum()/denom
                    else:
                        loss = _loss.sum()
                    losses.append(loss)
                    #distlosses.append(distloss(weights, t_samples)) # used for mip-nerf 360
                    distlosses.append(lossfun_distortion(weights, t_samples)) # used for mip-nerf 360
                # The loss is a sum of coarse and fine MSEs
                coarse_loss = losses[0]
                fine_loss = losses[-1]
                if args.two_bounce.loss.use_distloss:
                    coarse_loss += distlosses[0]
                    fine_loss += distlosses[-1]
                loss = args.two_bounce.loss.coarse_loss_mult * coarse_loss + fine_loss

                ret_i['loss'] = loss.unsqueeze(0)
                ret_i['loss_coarse'] = losses[0].unsqueeze(0)
                ret_i['loss_fine'] = losses[-1].unsqueeze(0)
                # ret_i['loss_dist_coarse'] = distlosses[0].unsqueeze(0)
                # ret_i['loss_dist_fine'] = distlosses[-1].unsqueeze(0)
            if detailed_output:
                # ret_i['ret'] = ret
                ret_i['c_distance'] = c_distance.reshape(1, -1, 1)
                ret_i['f_acc'] = f_acc.reshape(1, -1)
                ret_i['f_weights'] = f_weights.unsqueeze(0)
                ret_i['f_t_samples'] = f_t_samples.unsqueeze(0)

            return ret_i
        ###################

        # device = self.device
        # 1) Select Reflection model
        xs = render_kwargs['object_points'] # should be the ray termination point
        view_dirs = render_kwargs['view_dirs'] # should be the view dirs of the incoming ray 

        if not use_input_rays_as_query_rays:
            normal_vecs = render_kwargs['normal_vecs'] # should be the normal vector at point xs
            if self.train_type == '5dreflected_only':
                rays_o, rays_d = get_reflected_ray(xs, view_dirs, normal_vecs)
            else:
                raise NotImplementedError("Choose from {}".format(self.all_types))
        else:
            rays_o, rays_d = xs, view_dirs

        ret = {}
        batched = True
        device = rays_o.device
        if batched:
            DIM_BATCHIFY = 1
            B = rays_d.shape[0]  # batch_size
            flat_vec_shape = [B, -1, 3]
        else:
            DIM_BATCHIFY = 0
            flat_vec_shape = [-1, 3]

        # print("trainer.forward, rays, B", rays_o.shape, B) 
        # [B, N, pts, 3] -> [B, N*pts, 3]
        rays_o = torch.reshape(rays_o, flat_vec_shape).float()
        rays_d = torch.reshape(rays_d, flat_vec_shape).float()
        if compute_loss:
            gt_spec_radiance = render_kwargs['est_specular_radiance']
            gt_spec_radiance = torch.reshape(gt_spec_radiance, flat_vec_shape)

        # 2/3) Render rays and compute losses
        if compute_loss: 
            if self.to_loss_space is None: 
                if args.two_bounce.training.loss_space == 'linear':
                    self.to_loss_space = lambda x : x
                elif args.two_bounce.training.loss_space == 'srgb':
                    self.to_loss_space = lambda x : linear_rgb_to_srgb(x)
                else:
                    raise Exception(f'Invalid loss space {args.two_bounce.training.loss_space}') 
            if self.loss_fn is None: 
                if args.two_bounce.training.loss_type == 'l1':
                    self.loss_fn = lambda x,y : F.l1_loss(self.to_loss_space(x), self.to_loss_space(y), reduction='none')
                elif args.two_bounce.training.loss_type == 'mse':
                    # print('using mse')
                    self.loss_fn = lambda x,y : F.mse_loss(self.to_loss_space(x), self.to_loss_space(y), reduction='none')
                elif args.two_bounce.training.loss_type == 'weighted_mse':
                    #Same as rawNeRF https://arxiv.org/pdf/2111.13679.pdf
                    self.loss_fn = lambda x,y : F.mse_loss(self.to_loss_space(x), self.to_loss_space(y), reduction='none')\
                                            /(self.to_loss_space(x.detach())+1e-3)**2

        for i in tqdm(range(0, rays_o.shape[DIM_BATCHIFY], rayschunk), disable=not show_progress):
            # assumed ray is shape [B, npts, 3]
            # batchifying the first dimension
            ######
            if compute_loss:
                # print("rays_o: ", rays_o.shape, rays_o[:, i:i+rayschunk].shape)
                # print("rays_d: ", rays_d.shape, rays_d[:, i:i+rayschunk].shape)
                # print("gt_spec_radiance: ", gt_spec_radiance.shape, gt_spec_radiance[:, i:i+rayschunk].shape)
                ret_i = batchify_rays(rays_o[:, i:i+rayschunk], 
                                      rays_d[:, i:i+rayschunk],
                                      gt_spec_radiance[:, i:i+rayschunk], self.loss_fn
                                  )
            else:
                ret_i = batchify_rays(rays_o[:, i:i+rayschunk], 
                                  rays_d[:, i:i+rayschunk],
                                  None, None
                                  )
            ######
            for k, v in ret_i.items():
                if k not in ret:
                    ret[k] = []
                ret[k].append(v)

        extras = {}
        for k, v in ret.items():
            # print(v[-1].shape)
            # print("-->", k)
            if 'loss' in k:
                ret[k] = torch.cat(v, 0)
            else:
                ret[k] = torch.cat(v, DIM_BATCHIFY)
            # print(k, ret[k].shape)
            if 'loss' not in k: 
                extras[k] = ret[k]
        ####
        if detailed_output:
            # add reflected rays 
            extras['rays_d'] = rays_d
        ####
        losses = OrderedDict()
        if compute_loss:
            losses["total"] = ret['loss'].sum()
            losses["loss_coarse"] = ret['loss_coarse'].sum()
            losses["loss_fine"] = ret['loss_fine'].sum()
            # losses["loss_dist_coarse"] = ret['loss_dist_coarse'].sum()
            # losses["loss_dist_fine"] = ret['loss_dist_fine'].sum()

        return OrderedDict(
            [('losses', losses),
             ('extras', extras)])
