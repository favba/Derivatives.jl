using Derivatives, InplaceRealFFTW
using LESFilter: rfftfreq, fftfreq
using Base.Test

#=
Test function: f = sin(3x)*cos(3y)*sin(4z)
=#

lx = 1.0
ly = 1.0
lz = 1.0

nx = 32
ny = 32
nz = 32

x = reshape(linspace(0,lx*2π*(1-1/nx),nx),(nx,1,1))
y = reshape(linspace(0,ly*2π*(1-1/ny),ny),(1,ny,1))
z = reshape(linspace(0,lz*2π*(1-1/nz),nz),(1,1,nz))

field = @. sin(3*x) * cos(3*y) * sin(4*z) 

correct(f::typeof(Derivatives.dx)) = @. 3*cos(3*x) * cos(3*y) * sin(4*z) 
correct(f::typeof(Derivatives.dy)) = @. sin(3*x) * -3*sin(3*y) * sin(4*z)
correct(f::typeof(Derivatives.dz)) = @. sin(3*x) * cos(3*y) * 4*cos(4*z)

correct2(f::typeof(Derivatives.dx)) = @. -9*sin(3*x) * cos(3*y) * sin(4*z) 
correct2(f::typeof(Derivatives.dy)) = @. sin(3*x) * -9*cos(3*y) * sin(4*z)
correct2(f::typeof(Derivatives.dz)) = @. sin(3*x) * cos(3*y) * -16*sin(4*z)

correct3(f::typeof(Derivatives.dx)) = @. -27*cos(3*x) * cos(3*y) * sin(4*z) 
correct3(f::typeof(Derivatives.dy)) = @. sin(3*x) * 27*sin(3*y) * sin(4*z)
correct3(f::typeof(Derivatives.dz)) = @. sin(3*x) * cos(3*y) * -64*cos(4*z)


for (c,n) in zip((correct,correct2,correct3),(1,2,3))
  for (f,l) in zip((Derivatives.dx,Derivatives.dy,Derivatives.dz),(lx,ly,lz))
    @test f(field,l,n) ≈ c(f)
  end
end

field2 = PaddedArray(field)

for (c,n) in zip((correct,correct2,correct3),(1,2,3))
  for (f,l) in zip((Derivatives.dx,Derivatives.dy,Derivatives.dz),(lx,ly,lz))
    @test real(f(field,l,n)) ≈ c(f)
  end
end

rfft!(field2)

@inferred Derivatives.loopdx!(complex(field2),rfftfreq(nx,lx))
@inferred Derivatives.loopdy!(complex(field2),fftfreq(ny,ly))
@inferred Derivatives.loopdz!(complex(field2),fftfreq(nz,lz))