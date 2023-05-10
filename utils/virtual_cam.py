import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
# from scipy.optimize import least_squares

dot = lambda x,y: (x*y).sum(-1, keepdim=True)
norm = lambda x: torch.sqrt((x**2).sum(-1, keepdim=True))
transp = lambda x: x.transpose(-2,-1)
to_np = lambda x: x[0,0].detach().cpu().numpy()
normalize = lambda x: x/(1e-7+norm(x))

def get_virtual_cone_params_sphere(o,d,r,up,t,n,R):
    # Add singleton dimensions to given quantities for cone rays
    add_sing = lambda x: x[:,:,None,:]
    o_s  = add_sing(o)  # NxP x 1x3
    d_s  = add_sing(d)  # NxP x 1x3
    r_s  = add_sing(r)  # NxP x 1x3
    up_s = add_sing(up) # NxP x 1x3
    t_s = add_sing(t)   # NxP x 1x1
    n_s = add_sing(n)   # NxP x 1x3
    R_s = add_sing(R)   # NxP x 1x1
    
    ########### Compute primary cone rays
    d_cone,t_cone = get_primary_cone_rays(o_s,t_s,d_s,r_s,up_s)
    # Intersection of primary cone rays and oscullating sphere
    p_s = o_s + t_s*d_s
    #Find the center of osculating sphere by backtracing surface normal till R
    c_est = p_s - R_s*n_s
    # append primary ray point to cone rays
    p_int = intersect_ray_sphere(o_s,d_cone,c_est,R_s)
    t_int = norm(p_int-o_s)
    # nan_mask corresponding to rays that do not intersect
    nan_mask = torch.isnan(t_int)

    p_int = torch.cat([p_int, p_s],-2)
    t_int = torch.cat([t_int,t_s],-2)
    d_cone = torch.cat([d_cone, d_s],-2)
    # Primary intersection is always valid
    nan_mask = torch.cat([nan_mask, 
                          torch.zeros_like(nan_mask[:,:,[0]])],-2)

    ########## Compute reflected cone rays
    n_int = normalize(p_int-c_est)
    d_r = d_cone - 2*dot(d_cone, n_int)*n_int

    
    ######### Compute Cone params
    # 1. Virtual cone apex
    o_prime = closest_pt_to_lines(p_int, d_r)

    # 2. Virtual cone axis
    d_ro = normalize(p_int-o_prime)
    t_ro = norm(p_int-o_prime)
    d_ro_center = d_ro[:,:,[4]]
    # 3. Virtual cone radius
    # r_prime = norm(d_ro[:,:,:4]-d_ro_center).nansum(2,keepdim=True)/\
    #             (~nan_mask[:,:,:4]).sum(2,keepdim=True)
    # _min = 1./torch.sqrt(torch.tensor(12.)) 1./sqrt(12) too large
    _min = torch.tensor(1e-4)
    # all_nan_mask = nan_mask[:,:,:4,:].sum(-2) == 4
    # pdb.set_trace()
    r_prime = torch.nan_to_num(norm(d_ro[:,:,:4]-d_ro_center), 0.).sum(2,keepdim=True)/\
                ((~nan_mask[:,:,:4]).sum(2,keepdim=True) + 1e-7)
    r_prime = torch.clamp(r_prime, _min, 2.)
    # Remove singleton dimensions for cone rays
    rem_sing = lambda x: x[:,:,0]
    o_prime_r = rem_sing(o_prime)
    d_ro_center = rem_sing(d_ro_center)
    r_prime_r = rem_sing(r_prime)

    
    debug = 0
    if debug:
        import pdb; pdb.set_trace()
        # Considering slicing the image row by row when using render_view
        # With downsample 5, row size is 101 So 50 is the center value
        r_prime_nan = torch.nan_to_num(norm(d_ro[:,:,:4]-d_ro_center[:,:,None]), 0.).sum(2,keepdim=True)/4.
        plt.figure();plt.imshow(r_prime_nan[...,0,0]);plt.colorbar();plt.savefig('viz/r_prime_nan.png');plt.close()
        plt.figure();plt.imshow(R[...,0].clip(0,1).cpu());plt.colorbar();plt.savefig('viz/R.png');plt.close()
    
    return o_prime_r, d_ro_center, r_prime_r

def intersect_ray_sphere(o, d, c, R):
    # Sphere given by <x-c,x-c>=R**2
    # Ray given by x = o+td
    # Point of intersection t follows:
    # a2t^2 + a1t + a0 with following values
    a2 = 1.
    a1 = 2*dot(d, o-c)
    a0 = norm(o-c)**2 - R**2
    # Using quadratic formula finding the smaller of the two roots
    t_int = (-a1 - torch.sqrt(a1**2 - 4*a2*a0))/(2*a2)

    return o+t_int*d

def get_uv(N,up):
    u = normalize(torch.cross(N,up))
    v = normalize(torch.cross(N,u))
    return u, v

def get_primary_cone_rays(o, t, d, r, up):
    thetas = torch.Tensor([0.,np.pi/2,np.pi,3*np.pi/2])\
                [None,None,:,None]\
                .to(t.device)
    u,v = get_uv(d,up)
    p_u = r*t*torch.cos(thetas)
    p_v = r*t*torch.sin(thetas)
    p_d = t
    p_cone = o+(p_u*u + p_v*v + p_d*d)
    d_cone = normalize(p_cone-o)
    t_cone = norm(p_cone-o)
    return d_cone,t_cone

def closest_pt_to_lines(p_int, d_r):
    # Written in terminology described in the link
    w_i = p_int[...,None] # N x P x 5 x 3 x 1
    u_i = d_r[...,None] # N x P x 5 x 3 x 1
    I = torch.eye(3).to(p_int.device).float()
    # Add singleton based on batch dimensions of w_i
    I = I.reshape((1,)*(w_i.ndim-2)+I.shape) # N x P x 5 x 3 x 3
    A_i = I - u_i@transp(u_i) # N x P x 5 x 3 x 3
    p_i = A_i @ w_i # N x P x 5 x 3 x 1

    # nan_mask to exclude cone rays that do not intersect sphere
    # Sum along cone rays
    A = (A_i).nansum(-3, keepdim=True) # N x P x 1 x 3 x 3
    p = (p_i).nansum(-3, keepdim=True) # N x P x 1 x 3 x 1

    # try:
        # A_pinv = torch.inverse(A)
    A_pinv = torch.linalg.pinv(A)
    # except Exception as e: 
        # import pdb
        # pdb.set_trace()
    o_prime = A_pinv@p # N x P x 1 x 3 x 1

    return o_prime[...,0] # N x P x 1 x 3 x 1

def find_closest_point(pts, view_dirs):
    """
    From: https://math.stackexchange.com/questions/2598811/calculate-the-point-closest-to-multiple-rays
    Determine the closest point to an arbitrary number of rays, and optionally plot the results
 
    :param rays:    list of ray tuples (S, D) where S is the starting point & D is a unit vector
    :return:        scipy.optimize.OptimizeResult object from scipy.optimize.least_squares call
    """

    # Generate a starting position, the dimension-wise mean of each ray's starting position
    starting_P = torch.mean(pts, axis=0)

    # Start the least squares algorithm, passing the list of rays to our error function
    ans = least_squares(distance, starting_P, kwargs={'pts': pts, 'view_dirs': view_dirs})
    return ans


def distance(P, pts, view_dirs):
    """
    Calculate the distance between a point and each ray

    :param P:       1xd array representing coordinates of a point
    :param rays:    list of ray tuples (S, D) where S is the starting point & D is a unit vector
    :return:        nx1 array of closest distance from point P to each ray in rays
    """

    dims = 3 # we are in a 3D world 
    PRINT_DEBUG = False

    # Generate array to hold calculated error distances
    errors = torch.full([len(pts)*dims], torch.inf)
    P = torch.tensor(P).float()

    # For each ray, calculate the error and put in the errors array
    for i in range(len(pts)):
        S, D = pts[i], view_dirs[i]
        # if PRINT_DEBUG:
        #     print("i: {} --> ".format(i), S, D, P)
        t_P = D.dot((P - S).T)/(D.dot(D.T))
        if t_P > 0:
            errors[i*dims:(i+1)*dims] = (P - (S + t_P * D)).T
        else:
            errors[i*dims:(i+1)*dims] = (P - S).T

    # Convert the error array to a vector (vs a nx1 matrix)
    return errors.ravel()


if __name__ == '__main__':

    dict_ = torch.load('found_an_error_with_these_inputs_1666188473.2606776.pt', map_location=torch.device('cpu'))
    import pdb 
    pdb.set_trace()
    print(dict_['r'])
    virtual_pts, virtual_dirs, virtual_radii = get_virtual_cone_params_sphere(
        dict_['o'], dict_['d'], dict_['r'], dict_['up'], dict_['t'], dict_['n'], dict_['R']
    )
    print("is nan:", torch.isnan(virtual_pts).any(),torch.isnan(virtual_dirs).any(), \
                torch.isnan(virtual_radii).any(), len(torch.isnan(virtual_radii)))
    virtual_bundle = torch.cat([virtual_dirs, virtual_radii], dim=-1)