import torch
import numpy as np
import torch.nn as nn

# broadcast 2x3 affine matrix to 3x3
def broadcast( mat ):
    if isinstance(mat, np.ndarray ):
        ret = np.eye( 3, m=3, dtype=mat.dtype )
        ret[0:2,0:3]=mat
        return ret
    elif isinstance(mat, torch.Tensor ):
        ret = torch.unsqueeze( torch.eye( 3, m=3 ), dim=0 )
        ret[0,0:2,0:3]=mat
        return ret
    else:
        raise Exception("error")

# truncate 3x3 matrix into 2x3 
def to_affine( mat ):
    if isinstance(mat, torch.Tensor):
        return mat[:,0:2,0:3].clone()
    elif isinstance(mat,np.array):
        return mat[0:2,0:3]
    else:
        raise Exception("error")

# get 2x3 matrix
def get_matrix( angle, scalex, scaley, dx, dy, x, y ):
    alpha = scalex * torch.cos(angle)
    beta = scaley * torch.sin(angle)
    mat = torch.zeros( (2,3), requires_grad=True ).clone()
    mat[0,0]=alpha
    mat[0,1]=beta
    mat[0,2]=(1.0-alpha) * x - beta * y + dx
    mat[1,0]=-beta
    mat[1,1]=alpha
    mat[1,2]=beta * x + (1.0 - alpha ) * y + dy
    return torch.unsqueeze( mat, dim=0 )

def da( angle, scalex, scaley, dx, dy, x, y ):
    alpha = -scalex * torch.sin(angle)
    beta = scaley * torch.cos(angle)
    mat = torch.zeros( (2,3), requires_grad=True ).clone()
    mat[0,0]=alpha
    mat[0,1]=beta
    mat[0,2]= -alpha * x - beta * y
    mat[1,0]=-beta
    mat[1,1]=alpha
    mat[1,2]= beta * x - alpha * y
    return torch.unsqueeze( mat, dim=0 )

# isotropic scale
def ds( angle, scale, dx, dy, x, y ):
    alpha = torch.cos(angle)
    beta = torch.sin(angle)
    mat = torch.zeros( (2,3), requires_grad=True ).clone()
    mat[0,0]=alpha
    mat[0,1]=beta
    mat[0,2]= -alpha * x - beta * y
    mat[1,0]=-beta
    mat[1,1]=alpha
    mat[1,2]= beta * x - alpha * y
    return torch.unsqueeze( mat, dim=0 )

def dsx( angle, scalex, scaley, dx, dy, x, y ):
    alpha = torch.cos(angle)
    mat = torch.zeros( (2,3), requires_grad=True ).clone()
    mat[0,0]=alpha
    mat[0,2]=(1.0-alpha) * x
    mat[1,1]=alpha
    mat[1,2]=(1.0 - alpha ) * y
    return torch.unsqueeze( mat, dim=0 )

def dsy( angle, scalex, scaley, dx, dy, x, y ):
    beta = torch.sin(angle)
    mat = torch.zeros( (2,3), requires_grad=True ).clone()
    mat[0,1]=beta
    mat[0,2]=-beta * y
    mat[1,0]=-beta
    mat[1,2]=beta * x
    return torch.unsqueeze( mat, dim=0 )

def dx( angle, scalex, scaley, dx, dy, x, y ):
    mat = torch.zeros( (2,3), requires_grad=True ).clone()
    mat[0,2]=1.0
    return torch.unsqueeze( mat, dim=0 )

def dy( angle, scalex, scaley, dx, dy, x, y ):
    mat = torch.zeros( (2,3), requires_grad=True ).clone()
    mat[1,2]=1.0
    return torch.unsqueeze( mat, dim=0 )

# normalize matrix from pixel dimensions to edge based [-1,1]
def normalize_matrix( size ):
    w = float(size[0])
    h = float(size[1])
    mat = torch.eye( 3, m=3 )
    mat[0,0]=2.0/w
    mat[0,2]=1.0/w-1.0
    mat[1,1]=2.0/h
    mat[1,2]=1.0/h-1.0
    torch.unsqueeze( mat, dim=0 )
    return mat

# nearest, bilinear, bicubic, & 'zeros', 'border', 'reflection'
def warp_affine( inp, M, size, interpolation, padding ):
    if not isinstance( inp, torch.Tensor ):
        raise Exception("error")
    if not isinstance( M, torch.Tensor ):
        raise Exception("error")
    if M.dim()!=3:
        raise Exception("error")
    N = normalize_matrix( size )
    if M.shape[1]==2 and M.shape[2]==3:
        tf = torch.matmul( N, torch.matmul( broadcast(M), torch.linalg.inv( N ) ) )
    elif M.shape[1]==3 and M.shape[2]==3:
        tf = torch.matmul( N, torch.matmul( M, torch.linalg.inv( N ) ) )
    else:
        raise Exception("error")
    grid = nn.functional.affine_grid( tf[:,0:2,0:3], inp.shape, align_corners=False ).to( inp.device )
    res = nn.functional.grid_sample( inp, grid, mode=interpolation, padding_mode=padding, align_corners=False)
    return res
