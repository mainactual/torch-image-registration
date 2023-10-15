import torch

class PerPixelLoss:
    def __init__(self ):
        pass
    # @staticmethod
    # def loss_function( x, y ):
        # raise Exception("error")
    
    # parameters x, y are moving and fixed image respectively
    # returns a loss calculated for each pixel separately
    # to get scalar value, use torch.sum( loss ) 
    # if gradient=True, returns a gradient calculated for each pixel separately
    def forward( self, x, y, gradient=False ):
        raise Exception("error")
        # it would be fancy, if autograd worked for matrix data 
        # x_ = x.detach()
        # x_.requires_grad=True
        # loss_value = self.loss_function( x_, y )
        # loss_value.backward()
        # return loss_value.detach(), x_.grad


class PerPixel_L2Loss( PerPixelLoss ):
    def __init__(self):
        super().__init__()
    def forward( self, x, y, gradient=False ):
        if x.dim()!=4:
            raise Exception("error")
        loss = 0.5 * ( torch.pow( x - y, 2.0 ) )
        if not gradient:
            return loss
        grad=x-y
        return loss, grad

