# Utils for visualization notebooks
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np

def plot3d_sphere(ax, R, c, N=50):
    u = np.linspace(0, 2*np.pi, N)
    v = np.linspace(0,np.pi,N)
    x = c[0]+R[0]*np.outer(np.cos(u),np.sin(v))
    y = c[1]+R[0]*np.outer(np.sin(u),np.sin(v))
    z = c[2]+R[0]*np.outer(np.ones(np.size(u)),np.cos(v))
    ax.plot_surface(x,y,z,color='k',
                    linewidth=0.0,cstride=1, rstride=1, alpha=0.3)
# draw a vector

#https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot3d_arrow(ax,o,t,d, color="k",label=""):
    a = Arrow3D([o[0], (o + t*d)[0]], 
                [o[1], (o + t*d)[1]],
                [o[2], (o + t*d)[2]],
                mutation_scale=10,
                lw=1, arrowstyle="-|>", 
                color=color,
                label=label)
    ax.add_artist(a)

def plot3d_point(ax, p, label='', marker='.', color='b'):
    ax.scatter(p[0], p[1], p[2], label=label, marker=marker, color=color)

norm = lambda x: np.sqrt((x**2).sum(-1, keepdims=True))
normalize = lambda x: x/(1e-7+norm(x))
def get_uv(d, up = np.array([0.,1.,0.])):
    u = normalize(np.cross(d,up))
    v = normalize(np.cross(d,u))
    return u, v
    
def plot3d_cone(ax, o, t_max, d, r, 
                color='b',N=50): 
    t = np.linspace(0,t_max,N) 
    theta = np.linspace(0,2*np.pi, N)
    u,v = get_uv(d)
    p_u = r[0]*np.outer(t,np.cos(theta)) # Nx N
    p_v = r[0]*np.outer(t,np.sin(theta)) # N x N 
    p_d =  np.outer(t,np.ones(N)) #N x N
    p_xyz = o[None,None, :] + \
            p_u[...,None]*u[None,None,:] + \
            p_v[...,None]*v[None,None,:] + \
            p_d[...,None]*d[None,None,:]
    x, y, z = p_xyz[...,0], p_xyz[...,1], p_xyz[...,2]
    ax.plot_surface(x,y,z,color=color,linewidth=0.0,
                    cstride=1, rstride=1, alpha=0.3)
    