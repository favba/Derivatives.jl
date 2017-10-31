__precompile__()
module Derivatives

using InplaceRealFFTW
using LESFilter: rfftfreq, fftfreq

export dx, dy, dz, dx!, dy!, dz!

function loopdx!(fieldhat::AbstractArray{<:Complex,3}, kim::AbstractVector) 
  nx,ny,nz = size(fieldhat)
  Threads.@threads for l = 1:nz
    for j = 1:ny
      @simd for i = 1:nx
      @fastmath @inbounds fieldhat[i,j,l] = fieldhat[i,j,l]*kim[i]
      end
    end
  end
end

function dx(field::AbstractArray{T,3},len::Real,n::Integer=1) where T<:Union{Float32,Float64}
  nx,ny,nz = size(field)

  fieldhat = rfft(field,1)
  @fastmath kim = (rfftfreq(nx,len) .* im) .^ n

  loopdx!(fieldhat,kim)

  return irfft(fieldhat,nx,1)
end

function loopdy!(fieldhat::AbstractArray{<:Complex,3}, kim::AbstractVector) 
  nx,ny,nz = size(fieldhat)
  Threads.@threads for l = 1:nz
    for j = 1:ny
      @simd for i = 1:nx
        @fastmath @inbounds fieldhat[i,j,l] = fieldhat[i,j,l]*kim[j]
      end
    end
  end
end

function dy(field::AbstractArray{T,3},len::Real,n::Integer=1) where T<:Union{Float32,Float64}
  nx,ny,nz = size(field)

  fieldhat = rfft(field,2)

  @fastmath kim = (rfftfreq(ny,len) .* im) .^ n

  loopdy!(fieldhat,kim)

  return irfft(fieldhat,ny,2)
end

function loopdz!(fieldhat::AbstractArray{<:Complex,3}, kim::AbstractVector) 
  nx,ny,nz = size(fieldhat)
  Threads.@threads for l = 1:nz
    for j = 1:ny
      @simd for i = 1:nx
        @fastmath @inbounds fieldhat[i,j,l] = fieldhat[i,j,l]*kim[l]
      end
    end
  end
end

function dz(field::AbstractArray{T,3},len::Real,n::Integer=1) where T<:Union{Float32,Float64}
  nx,ny,nz = size(field)

  fieldhat = rfft(field,3)

  @fastmath kim = (rfftfreq(nz,len) .* im) .^ n

  loopdz!(fieldhat,kim)

  return irfft(fieldhat,nz,3)
end

function dx!(field::AbstractPaddedArray{T,3,L},len::Real,n::Integer=1) where {T<:Union{Float64,Float32},L}

  nx,ny,nz = size(real(field))

  rfft!(field,1)

  @fastmath kim = (rfftfreq(nx,len) .* im) .^ n

  loopdx!(complex(field),kim)

  return irfft!(field,1)
end

function dx(field::AbstractPaddedArray{T,3,L},len::Real,n::Integer=1) where {T<:Union{Float64,Float32},L}
  fieldhat = copy(field)
  return dx!(fieldhat,len,n)
end

function dy!(field::AbstractPaddedArray{T,3,L},len::Real,n::Integer=1) where {T<:Union{Float64,Float32},L}

  nx,ny,nz = size(real(field))
  rfft!(field,1:2)

  @fastmath kim = (fftfreq(ny,len) .* im) .^ n

  loopdy!(complex(field),kim)

  return irfft!(field,1:2)
end

function dy(field::AbstractPaddedArray{T,3,L},len::Real,n::Integer=1) where {T<:Union{Float64,Float32},L}
  fieldhat = copy(field)
  return dy!(fieldhat,len,n)
end


function dz!(field::AbstractPaddedArray{T,3,L},len::Real,n::Integer=1) where {T<:Union{Float64,Float32},L}

  nx,ny,nz = size(real(field))
  rfft!(field,(1,3))

  @fastmath kim = (fftfreq(nz,len) .* im) .^ n

  loopdz!(complex(field),kim)

  return irfft!(field,(1,3))
end

function dz(field::AbstractPaddedArray{T,3,L},len::Real,n::Integer=1) where {T<:Union{Float64,Float32},L}
  fieldhat = copy(field)
  return dz!(fieldhat,len,n)
end


end # module
