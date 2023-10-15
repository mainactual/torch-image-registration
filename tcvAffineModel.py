import torch
from torch import nn
import numpy as np
import tcvAffineTransform as Affy
import tcvMapping as Mapping

class AffineModel( nn.Module ):
    # cx, cy are the constant center of rotation and scale 
    # size is target image size as (width,height)
    # bordermode="reflection", "border","zeros"
    def __init__( self, cx, cy, bordermode ):
        super().__init__()
        self.cx=cx
        self.cy=cy
        self.bordermode=bordermode
        
        # fill these!
        self.p = nn.ParameterList()
        self.m = [Mapping.IdentityMapping() for i in range( len(self.p) )]
    
    #derive these!
    def get_matrix( self ):
        return Affy.get_matrix( 0.0, 1.0, 1.0, 0.0, 0.0, self.cx, self.cy )
    
    # custom to(device) function does not call "to" for the base class
    def to( self, device ):
        pass
    
    @staticmethod
    def safe_val( val ):
        if isinstance( val, torch.Tensor ):
            if val.shape != torch.Size([]):
                raise Exception("error")
            return val
        elif isinstance( val, float ):
            return torch.tensor( [val] )
        else:
            raise Exception("error")
    
    # set the f first, then solve to produce val for the matrix
    def set_parameter( self, idx, val, f = None ):
        if idx >= len(self.p):
            raise Exception("error")
        if f is not None:
            if not isinstance( f, Mapping.IdentityMapping ):
                raise Exception("error")
            self.m[idx] = f
        self.p[idx] = self.m[idx].solve( self.safe_val( val ) )
    
    def get( self, idx ):
        if idx >= len(self.p):
            raise Exception("error")
        return self.m[idx].value( self.p[idx] )
    
    def forward( self, x ):
        if x.dim()!=4:
            raise Exception("error")
        size = ( x.shape[3], x.shape[2] )
        s = Affy.warp_affine( x, self.get_matrix(), size, 'bicubic', self.bordermode)
        return s

