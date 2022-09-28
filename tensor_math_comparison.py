import torch
x=torch.tensor([1,2,3])
y=torch.tensor([9,8,7])

#Addition
z1=torch.empty(3)
torch.add(x,y,out=z1)
#print(z1)
z2=torch.add(x,y)
#print(z2)
z=x+y,
#print(z)

#Subtraction
z=x-y
#print(z)

#Division
z=torch.true_divide(x,y)
#print(z)

#inplace operation
t=torch.zeros(3)
t.add_(x)
t+=x;

#Exponentiation
z=x.pow(2)
z=x**2
#print(z)

#Simple Comparison
z=x<0
z=x>0
print(z)

#Matrix Multiplication
x1=torch.rand((2,5))
x2=torch.rand((5,3))
x3=torch.mm(x1,x2)
x3=x1.mm(x2)
#print(x3)

#Matrix exponentiation
matrix_exp=torch.rand(5,5)
matrix_exp.matrix_power(3)
#print(matrix_exp)

#Element wise multiplication
z=x*y,
#print(z)

#dot product
z=torch.dot(x,y)
#print(z)

#Batch matrix multiplication
batch=32
n=10
m=20
p=30

tensor1=torch.rand((batch, n, m))
tensor2=torch.rand((batch, m, p))
out_bmm=torch.bmm(tensor1,tensor2)
#print(out_bmm)

#Example of Broadcasting
x1=torch.rand((5,5))
x2=torch.rand((1,5))
z=x1-x2
z=x1**x2

#Other useful tensor operation
sum_x=torch.sum(x,dim=0)
values,indices=torch.max(x, dim=0)
values,indices=torch.min(x, dim=0)
abs_x=torch.abs(x)
z=torch.argmax(x, dim=0)
z=torch.argmin(x,dim=0)
mean_x=torch.min(x.float(),dim=0)
z=torch.eq(x,y)
sorted_y, indices=torch.sort(y,dim=0, descending=True)
z=torch.clamp(x,min=0, max=2)
x=torch.tensor([1,0,1,1,1], dtype=torch.bool)
z=torch.any(x)
z=torch.all(x)
print(z)


