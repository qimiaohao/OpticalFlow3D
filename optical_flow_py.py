import torch
from scipy.ndimage.filters import sobel, gaussian_filter
import numpy as np
from numpy import array
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import interpn
import scipy.ndimage
from enum import Enum
import scipy.signal
import scipy.sparse
from scipy.ndimage import zoom
import matplotlib.pyplot as plt



def imresize(input, scale):
    if scale < 1:
        step = int(1/scale)
        output = input[::step,::step,::step]
    if scale > 1:
        output = input
        for i in range(3):
            output = np.repeat(output, scale, axis=i)
    if scale == 1:
        output = input
    return output



class InterpolationMode(Enum):
    nearest = 1
    linear  = 2

class BoundaryMode(Enum):
    zero      = 1
    replicate = 2
    symmetric = 3
    periodic  = 4

def interp3d(x, y, z, image, bmode=BoundaryMode.replicate, lmode=InterpolationMode.linear):
    def tex(x, y, z):
        if bmode is BoundaryMode.zero:
            raise NotImplementedError()
        elif bmode is BoundaryMode.replicate:
            x = np.clip(x, 0, image.shape[0] - 1)
            y = np.clip(y, 0, image.shape[1] - 1)
            z = np.clip(z, 0, image.shape[2] - 1)
        elif bmode is BoundaryMode.symmetric:
            raise NotImplementedError()
        elif bmode is BoundaryMode.periodic:
            raise NotImplementedError()
        return image[x.flatten(), y.flatten(), z.flatten()].reshape(x.shape)

    if lmode is InterpolationMode.nearest:
        val = tex(np.round(x.astype(int)), np.round(y.astype(int)), np.round(z.astype(int)))
    elif lmode is InterpolationMode.linear:
        p = np.stack((x, y, z))
        p_floor = np.floor(p).astype(int)
        x0, y0, z0 = p_floor[0], p_floor[1], p_floor[2]
        s000 = tex(x0, y0, z0)
        s001 = tex(x0, y0, 1 + z0)
        s010 = tex(x0, 1 + y0, z0)
        s011 = tex(x0, 1 + y0, 1 + z0)
        s100 = tex(1 + x0, y0, z0)
        s101 = tex(1 + x0, y0, 1 + z0)
        s110 = tex(1 + x0, 1 + y0, z0)
        s111 = tex(1 + x0, 1 + y0, 1 + z0)

        w1 = p - p_floor
        w0 = 1. - w1
        val = (
                w0[0] * (
                w0[1] * (w0[2] * s000 + w1[2] * s001) +
                w1[1] * (w0[2] * s010 + w1[2] * s011)
        ) +
                w1[0] * (
                        w0[1] * (w0[2] * s100 + w1[2] * s101) +
                        w1[1] * (w0[2] * s110 + w1[2] * s111)
                )
        )
    return val

def LinearCG(A, b, x0, tol=1e-5, imax=5000):
    xk = x0
    rk = A(xk) - b
    pk = -rk
    rk_norm = np.linalg.norm(rk)

    num_iter = 0

    while num_iter < imax and rk_norm > tol:
        apk = A(pk)
        rkrk = np.sum(rk.reshape(-1,1)**2)

        alpha = rkrk / np.sum(np.reshape(pk*apk,(-1,1)))
        xk = xk + alpha * pk
        rk = rk + alpha * apk
        beta = np.sum(rk.reshape(-1,1)**2) / rkrk
        pk = -rk + beta * pk

        num_iter += 1

        rk_norm = np.linalg.norm(rk)
       # print('Iteration: {}  \t residual = {:.6f}'.
              #format(num_iter, rk_norm))
        if num_iter > imax:
            print ('reach the max iteration')

    return xk

def partial_deriv(o):
    N=o.ndim
    gt= o[:, :, :, 1] - o[:, :, :, 0]
    img1 = o[:,:,:, 0]
    grad_kern = np.array([1, 0, -1]) / 2
    gz = scipy.ndimage.convolve(img1, grad_kern[None, None, :], mode='nearest')
    gy = scipy.ndimage.convolve(img1, grad_kern[None, :, None], mode='nearest')
    gx = scipy.ndimage.convolve(img1, grad_kern[:, None, None], mode='nearest')
    gs = np.stack((gx,gy,gz),axis=N-1)
    return gt,gs

def solve_flow(o,priors,tol,iter):
    gt,gs=partial_deriv(o)
    Nz,Ny,Nz,Ndimqm=o.shape
    N=o.ndim
    alpha_f = lambda u: np.concatenate((priors[0]*u[:,:,:,0:2],np.expand_dims( priors[1]*u[:,:,:,2],axis=N-1)),axis=N-1)
    beta_f = lambda u: np.concatenate((priors[3]*u[:,:,:,0:2],np.expand_dims(priors[1]*u[:,:,:,2],axis=N-1)),axis=N-1)
    nabla2 = lambda u: scipy.ndimage.filters.laplace(u)
    D = lambda u: gs*np.expand_dims(np.sum(gs*u,N-1),axis=N-1)
    A = lambda u: D(u) - alpha_f(nabla2(u)) + beta_f(u)
    b = -gs*np.expand_dims(gt,axis=N-1)



    u = LinearCG(A,b, tol=tol, imax=iter,x0=np.zeros(b.shape))

    u = u.clip(-1,1)

    u = np.stack((scipy.signal.medfilt(u[:,:,:,0],kernel_size=3),scipy.signal.medfilt(u[:,:,:,1],kernel_size=3),scipy.signal.medfilt(u[:,:,:,2],kernel_size=3)),axis=N-1)

    return u


def optical_flow(o):


    pyramid_level=2
    priors = np.array([1e-4, 1e-7, 1e-4, 1e-7])
    tol = 1e-6
    iter = 1000


    dim_x=o.shape

    T=dim_x[-1]
    dim_x=dim_x[:-1]
    N=len(dim_x)  # 2d optical flow or 3d optical flow
    scale=2


    prev_vol=o[:,:,:,0]
    next_vol=o[:,:,:,1]
    o_pyramid=[]


    o_vol1= gaussian_filter(o, 1.5, mode='constant')
    o_vol2=gaussian_filter(o, 1.5, mode='constant')[::2, ::2, ::2]

    #prev_vol2 = gaussian_filter(prev_vol, 1.5, mode='constant')[::2, ::2, ::2]
    #next_vol2 = gaussian_filter(next_vol, 1.5, mode='constant')[::2, ::2, ::2]

    o_pyramid.append(o_vol1)
    o_pyramid.append(o_vol2)



    init_flow_size = o_pyramid[-1].shape
    init_flow_size=list(init_flow_size)
    init_flow_size[-1]=N
    flow = np.zeros(init_flow_size)
    warping_iters=2

    for idx in reversed(range(pyramid_level)):
        o=o_pyramid[idx]
        [xx, yy, zz] = np.meshgrid(np.arange(o.shape[1]),np.arange(o.shape[0]),np.arange(o.shape[2]))
        for j in range(warping_iters):
            if not (idx == (pyramid_level-1) and j == 0):
                o1_warp = interp3d( xx + flow[:, :, :, 1],yy + flow[:, :, :, 0],zz + flow[:, :, :, 2],o[:, :, :, 0])
                o_tmp=np.stack((o1_warp,o[:, :, :, 1]),axis=N)

                o_tmp[np.isnan(o_tmp)]=0

                o_tmp = np.stack((scipy.signal.medfilt(o_tmp[:,:,:,0],kernel_size=N),scipy.signal.medfilt(o_tmp[:,:,:,0],kernel_size=N)),axis=-1)

            else:
                o_tmp = o

            flow_delta = solve_flow(o_tmp, priors, tol, iter)

            mean_delta_u=np.mean(np.abs(flow_delta.reshape(-1,1)))

            print('-- warp'+str(j+1)+', mean(|\Delta flow|) = '+str(mean_delta_u))

            if mean_delta_u > 0.00:
                flow[:,:,:,0:2]=flow[:,:,:,0:2]+flow_delta[:,:,:,0:2]
                flow[:,:,:,2]=flow[:,:,:,2]+flow_delta[:,:,:,2]
            else:
                print('Insignificant \Delta flow; reject adding to total flow')
                break

        if  idx >0:
            flow_t=[]
            for tt in range (N):
                flow_t1=zoom(flow[:,:,:,tt],(np.round(scale**(idx)),np.round( scale**(idx)),np.round(scale**(idx))))
                flow_t.append(flow_t1)
            flow=np.stack(flow_t,axis=-1)


    return flow









