import torch
from torch import nn
import numpy as np
import cv2 as cv

import tcvMapping as Mapping
import tcvAffineTransform as Affy
import tcvAffineModel as AffyModel
import tcvAffineModelAnalytic as AffyModelAnal
import tcvPerPixelLoss as PerPixelLoss
import time

# convert a 8-bit RGB image [H,W,C] to tensor [B,C,H,W] 
def convert_to_tensor( mat ):
    if not isinstance( mat, np.ndarray ):
        raise Exception("error")
    
    if len(mat.shape)==2 or mat.shape[0]==1:
        img = torch.unsqueeze( torch.from_numpy( mat ), dim=0 )
    else:
        img = torch.permute( torch.from_numpy( mat ), (2,0,1) )
    return torch.unsqueeze( img.float()/127.5 - 1.0, dim=0 )

# convert a tensor [B,C,H,W] to 8-bit RGB [H,W,C]
def convert_to_numpy( mat ):
    if not isinstance( mat, torch.Tensor ):
        raise Exception("error")
    
    if mat.shape[1]==3:
        t = torch.permute( mat[0,:,:,:], (1,2,0) )
    else:
        t = mat[0,0,:,:]
    return np.clip( t.numpy()*127.5+127.5, 0.0, 255.0 ).astype( np.uint8 )

# define a Cauchy loss
class PerPixel_CauchyLoss( PerPixelLoss.PerPixelLoss ):
    def __init__(self):
        super().__init__()
    def forward( self, x, y, gradient = False ):
        if x.dim()!=4:
            raise Exception("error")
        c = torch.pow( x - y, 2.0 ) + 1.0
        loss = torch.log( c )
        if not gradient:
            return loss
        grad= (x - y)/c 
        return loss,grad

# four parameter affine transforms with angle, scale, dx, dy
class MyModel( AffyModel.AffineModel ):
    def __init__( self, cx, cy, bordermode ):
        super().__init__( cx, cy, bordermode )
        # angle, scalex, scaley, dx, dy
        self.p = nn.ParameterList( torch.tensor( [i] ) for i in [0.0, 0.0, 0.0, 1.0]  )
        self.m = [Mapping.IdentityMapping() for i in range( len(self.p) )]
    def get_matrix( self ):
        return Affy.get_matrix( self.get(0), self.get(3), self.get(3), self.get(1), self.get(2), self.cx, self.cy )

class MyModelAnalytic( AffyModelAnal.AffineModelAnalytic ):
    def __init__( self, cx, cy, bordermode, movingImage ):
        super().__init__( cx, cy, bordermode, movingImage )
        # angle, scalex, scaley, dx, dy
        self.p = nn.ParameterList( torch.tensor( [i] ) for i in [0.0, 0.0, 0.0, 1.0] )
        self.m = [Mapping.IdentityMapping() for i in range( len(self.p) )]
    def get_matrix( self ):
        return Affy.get_matrix( self.get(0), self.get(3), self.get(3), self.get(1), self.get(2), self.cx, self.cy )
    def get_derva( self, idx ):
        isy=3
        if idx==0:
            return Affy.da( self.get(0),self.get(3),self.get(isy),self.get(1),self.get(2), self.cx, self.cy )
        elif idx==1:
            return Affy.dx( self.get(0),self.get(3),self.get(isy),self.get(1),self.get(2), self.cx, self.cy )
        elif idx==2:
            return Affy.dy( self.get(0),self.get(3),self.get(isy),self.get(1),self.get(2), self.cx, self.cy )
        elif idx==3:
            return Affy.ds( self.get(0),self.get(3), self.get(1),self.get(2), self.cx, self.cy )
        else:
            raise Exception("error")

def registration( source_, target_ ):
    
    cx = float( source_.shape[1] )/2.0
    cy = float( source_.shape[0] )/2.0
    size = (source_.shape[0], source_.shape[1])
    source = convert_to_tensor( source_ )
    target = convert_to_tensor( target_ )
    
    learning_rate = 0.1
    
    loss_func = PerPixel_CauchyLoss()

    model=None
    if True:
        model = MyModel( cx, cy, "zeros" )
        model.set_parameter( 0, 0.0, Mapping.SigmoidMapping( -0.25, 0.25 ) )
        model.set_parameter( 1, 0.0, Mapping.SigmoidMapping( -80.0, 80.0 ) )
        model.set_parameter( 2, 0.0, Mapping.SigmoidMapping( -130.0, 130.0 ) )
        model.set_parameter( 3, 1.0, Mapping.SigmoidMapping(0.5, 1.6) )
        model.train()
        optim = torch.optim.Adam( model.parameters(), lr = learning_rate)
    
    model2=None
    if True:
        model2 = MyModelAnalytic( cx, cy, "zeros", source )
        model2.loss_func=loss_func
        model2.set_parameter( 0, 0.0, Mapping.SigmoidMapping( -0.25, 0.25 ) )
        model2.set_parameter( 1, 0.0, Mapping.SigmoidMapping( -80.0, 80.0 ) )
        model2.set_parameter( 2, 0.0, Mapping.SigmoidMapping( -130.0, 130.0 ) )
        model2.set_parameter( 3, 1.0, Mapping.SigmoidMapping(0.5, 1.6) )
        model2.train()
        optim2 = torch.optim.Adam( model2.parameters(), lr = learning_rate )
    
    t1=0.0
    t2=0.0
    for i in range(0,200):
        if model is not None:
            s = model( source )
            loss = torch.sum( loss_func.forward( s, target, False ) )
            loss.backward()
            optim.step()
            optim.zero_grad()
        
        if model2 is not None:
            s2,loss2 = model2( source, target )
            optim2.step()
            optim2.zero_grad()
        
        if model is not None:
            retva = convert_to_numpy( s.detach() )
            cv.imshow( "autograd", cv.absdiff( retva, target_ ) )
        if model2 is not None:
            retva2 = convert_to_numpy( s2.detach() )
            cv.imshow( "analytic", cv.absdiff( retva2, target_ ) )
        
        if cv.waitKey( 1 )==ord('q'):
            break
    
    cv.waitKey( 0 )




sm=(256,384)

target = cv.imread( 'a.jpg' )
target = cv.resize(target,sm,0.0,0.0,cv.INTER_LINEAR )
target = cv.cvtColor( target, cv.COLOR_BGR2GRAY )

source = cv.imread( 'b.jpg' )
source=cv.resize(source,sm,0.0,0.0,cv.INTER_LINEAR )
source = cv.cvtColor( source, cv.COLOR_BGR2GRAY )

registration( source, target )
