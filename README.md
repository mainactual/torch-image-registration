# torch-image-registration

A traditional image registration employs chain rule for calculating derivatives

$$ \frac{\partial}{\partial p}= C( F(\mathbf{x}),M( \mathbf{T} \mathbf{x}) )=C'(F(\mathbf{x}),M( \mathbf{T} \mathbf{x})) * \nabla M( \mathbf{T} \mathbf{x}) * \mathbf{T}'\mathbf{x}$$ 

where C is the cost function, F and M are fixed and moving image respectively and 

$$\mathbf{T}:=\mathbf{T}(p)$$ 

is an affine transform of parameters p. 

While automatic differentiation, e.g., torch.autograd [1], makes things a lot easier, the traditional method is still sound. And if the loss C is simply a sum over the image, the user only needs to provide its derivative. Consider for example the Cauchy-loss:

```python
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

loss_func = PerPixel_CauchyLoss()
```

The autograd model takes four parameters angle, displacement x and y and scale.
```python
class MyModel( AffyModel.AffineModel ):
    def __init__( self, cx, cy, bordermode ):
        super().__init__( cx, cy, bordermode )
        # angle, scalex, scaley, dx, dy
        self.p = nn.ParameterList( torch.tensor( [i] ) for i in [0.0, 0.0, 0.0, 1.0]  )
        self.m = [Mapping.IdentityMapping() for i in range( len(self.p) )]
    def get_matrix( self ):
        return Affy.get_matrix( self.get(0), self.get(3), self.get(3), self.get(1), self.get(2), self.cx, self.cy )
```
The analytic model also needs to calculate moving-image gradients once. They can be precalculated using a custom method, e.g., difference of Gaussian.
```python
class MyModelAnalytic( AffyModelAnal.AffineModelAnalytic ):
    def __init__( self, cx, cy, bordermode, movingImage ):
        super().__init__( cx, cy, bordermode, movingImage )
```
As well as derivatives of T
```python
def get_derva( self, idx ):
        if idx==0:
            return Affy.da( self.get(0),self.get(3),self.get(3),self.get(1),self.get(2), self.cx, self.cy )
        elif idx==1:
            return Affy.dx( self.get(0),self.get(3),self.get(3),self.get(1),self.get(2), self.cx, self.cy )
        elif idx==2:
            return Affy.dy( self.get(0),self.get(3),self.get(3),self.get(1),self.get(2), self.cx, self.cy )
        elif idx==3:
            return Affy.ds( self.get(0),self.get(3), self.get(1),self.get(2), self.cx, self.cy )
```

Then models are initiated using Sigmoid-function to map real numbers to a valid parameter range as
```python
model = MyModel( cx, cy, "zeros" )
model.set_parameter( 0, 0.0, Mapping.SigmoidMapping( -0.25, 0.25 ) )
model.set_parameter( 1, 0.0, Mapping.SigmoidMapping( -80.0, 80.0 ) )
model.set_parameter( 2, 0.0, Mapping.SigmoidMapping( -130.0, 130.0 ) )
model.set_parameter( 3, 1.0, Mapping.SigmoidMapping(0.5, 1.6) )
model.train()
optim = torch.optim.Adam( model.parameters(), lr = learning_rate)

model2 = MyModelAnalytic( cx, cy, "zeros", source )
model2.loss_func=loss_func
# ...similar for the model2
optim2 = torch.optim.Adam( model2.parameters(), lr = learning_rate )
```
The training loop for the autograd model looks familiar:
```python
s = model( source )
loss = torch.sum( loss_func.forward( s, target, False ) )
loss.backward()
optim.step()
optim.zero_grad()
```
And the analytic model simply misses the `backward` step as the gradients are evaluated forward already.
```python
s2,loss2 = model2( source, target )
optim2.step()
optim2.zero_grad()
```
`loss2` returned by `forward` is sum of per pixel loss calculated inside the function.

As a result, one can observe that from image to image, either analytic or autograd converges better. Many times, analytic converges faster if it starts to converge (L2 or Cauchy loss are only mediocre for many registration tasks). Python-implementation of analytic differentials is actually a bit slower than torch-autograd and that is due to fact of unnecessary permutes and caching of arrays. In real world scenario, they can be calculated in-place. Also, the loss-function need not be per pixel and, for example, normalized mutual information can be calculated using the dot-products inside `AffineModelAnalytic` in similar fashion.




[1] [pytorch-autograd](https://pytorch.org/docs/stable/autograd.html)

