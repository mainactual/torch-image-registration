import torch

class IdentityMapping:
    def __init__( self ):
        pass
    # maps input by a function y=f(x)
    def value( self, x ):
        return x
    # calculates df(x)/dx
    def derivative( self, x ):
        return torch.tensor( [1.0] )
    # inverse, return x=f^-1(y)
    def solve( self, x ):
        return x

class SigmoidMapping( IdentityMapping ):
    def __init__( self, low, high ):
        super().__init__()
        
        if low >= high:
            raise Exception("error")
        self.low=low
        self.high=high
    def value( self, x ):
        return self.low + (self.high-self.low)/( 1.0 + torch.exp( -x ))
    def derivative( self, x ):
        return (self.high-self.low) * torch.exp( -x )/torch.pow( 1.0 + torch.exp( -x ), 2.0 )
    def solve( self, x ):
        if x <= self.low or x >= self.high:
            raise Exception("error")
        return -torch.log( ( self.high - self.low )/( x - self.low ) - 1.0 )
