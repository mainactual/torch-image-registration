import torch
import torch.nn as nn
import numpy as np
import tcvAffineTransform as Affy
import tcvAffineModel as AffyModel

def my_sobel( x ):
    kernelx = torch.tensor( [ [-1.0, 0.0, 1.0],[ -2.0, 0.0, 2.0],[-1.0, 0.0, 1.0] ] )/8.0
    kernely = torch.transpose( kernelx, 0,1)
    weight = torch.unsqueeze( torch.stack( (kernelx, kernely), dim=0 ), dim=1 )
    z = nn.functional.pad( x, (1,1,1,1), mode='reflect' )
    y = nn.functional.conv2d( z, weight, padding='valid' )
    return y

class AffineModelAnalytic( AffyModel.AffineModel ):
    # cx, cy are the constant center of rotation and scale 
    # size is target image size as (width,height)
    # bordermode="reflection", "border","zeros"
    # if movingImage type is B,2,H,W, then it is expected to contain gradients
    def __init__( self, cx, cy, bordermode, movingImage ):
        super().__init__( cx, cy, bordermode )
        if movingImage.dim()!=4:
            raise Exception("error")
        if movingImage.shape[0]!=1:
            raise Exception("error")
        if movingImage.shape[1]==2:
            self.gradients = movingImage.detach()
        else:
            # calculate spatial gradients
            self.gradients = my_sobel( movingImage )
            # B,2,H,W 
        
        size = ( movingImage.shape[2], movingImage.shape[3] )
        # create the meshgrid, a constant
        # value in each pixel is (x,y,1)
        x_values, y_values = np.meshgrid( np.arange(size[0]), np.arange(size[1]) )
        v = np.stack( (x_values, y_values, np.ones( (size[1],size[0]) ) ),axis=-1 ).astype(np.float32)
        # H,W,3
        indices = torch.unsqueeze( torch.from_numpy( v ), axis=0 ) 
        # 1,H,W,3
        self.indices = torch.unsqueeze( indices.view(-1,3 ), axis=2 )
        self.loss_func=None
        
        # remember to fill these!
        #self.p = nn.ParameterList()
        #self.m = [Mapping.IdentityMapping() for i in range( len(self.p) )]
    
    #derive these!
    def get_matrix( self ):
        return Affy.get_matrix( 0.0, 1.0, 1.0, 0.0, 0.0, self.cx, self.cy )
    def get_derva( self, idx ):
        raise Exception("not impl")
    
    def to( self, device ):
        super().to(device)
        self.indices=self.indices.to(device)
        self.gradients=self.gradients.to(device)
    
    def forward( self, x ):
        raise Exception("error")
    
    def forward( self, x, y ):
        if self.loss_func is None:
            raise Exception("error")
        if x.dim()!=4:
            raise Exception("error")
        size = ( x.shape[3], x.shape[2] )
        
        # C( I(x), J( Mx ) ) => C'( I(x), J(Mx) ) * Grad(J(Mx)) dot M'x
        with torch.no_grad():
            device = x.device
            tform = self.get_matrix() 
            # bilinear, bicubic
            current_image = Affy.warp_affine( x, tform, size,'bicubic', self.bordermode )
            pixelloss, derva = self.loss_func.forward( current_image, y, True )
            
            current_gradients = Affy.warp_affine( self.gradients, tform, size, 'bicubic', "border")
            current_gradients = torch.permute( current_gradients, (0,2,3,1) ) # B,H,W,2
            current_gradients = current_gradients.view(-1,2 )
            
            loss = torch.sum( pixelloss )
            
            for i in range(len(self.p)):
                M = self.get_derva( i ).to(device)
                dot = torch.sum( current_gradients * torch.squeeze(torch.matmul( M, self.indices ),axis=2), dim=1 )
                t = torch.sum( derva * dot.reshape( x.shape ) )
                self.p[i].grad = torch.tensor([t.item()])
            
            return current_image, loss