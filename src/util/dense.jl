using CUDArt
import Base: similar, convert, copy, copy!, resize!, issparse
import Base: eltype, length, ndims, size, strides, stride, pointer, isempty, getindex, setindex!, sub
import Base: rand!, randn!, fill!
import CUDArt: to_host

### KUdense parametrized by array type, element type, and ndims:

type KUdense{A,T,N}; arr; ptr; end

### CONSTRUCTORS

KUdense(a)=KUdense{atype(a),eltype(a),ndims(a)}(a, reshape(a, length(a)))

KUdense{A<:Array,T,N}(::Type{A}, ::Type{T}, d::NTuple{N,Int})=KUdense(Array(T,d))
KUdense{A<:CudaArray,T,N}(::Type{A}, ::Type{T}, d::NTuple{N,Int})=KUdense(CudaArray(T,d))

convert(::Type{KUdense}, a)=KUdense(a)
convert{A<:BaseArray}(::Type{A}, a::KUdense)=convert(A, a.arr)
convert{A,B}(::Type{KUdense{B}}, a::A)=KUdense(convert(B, a))

similar{A,T,N}(a::KUdense{A}, ::Type{T}, d::NTuple{N,Int})=KUdense(A,T,d)
similar{A,T,N}(a::KUdense{A,T}, d::NTuple{N,Int})=KUdense(A,T,d)
similar{A,T,N}(a::KUdense{A,T,N})=KUdense(A,T,size(a))

arr(a::Vector,d::Dims)=pointer_to_array(pointer(a), d)
arr(a::CudaVector,d::Dims)=CudaArray(a.ptr, d, a.dev)

atype(::Array)=Array
atype(::SubArray)=Array
atype(::CudaArray)=CudaArray

### BASIC ARRAY OPS

atype{A}(::KUdense{A})=A

for fname in (:eltype, :length, :ndims, :size, :strides, :pointer, :isempty)
    @eval $fname(a::KUdense)=$fname(a.arr)
end

for fname in (:size, :stride)
    @eval $fname(a::KUdense,n)=$fname(a.arr,n)
end

for fname in (:getindex, :setindex!, :sub)
    @eval $fname(a::KUdense,n...)=$fname(a.arr,n...)
end

### BASIC COPY

copy!{A,B,T}(a::KUdense{A,T}, b::KUdense{B,T})=(resize!(a, size(b)); copy!(a.arr, 1, b.arr, 1, length(b)); a)
copy!{A,T}(a::KUdense{A,T}, b::Union(Array{T},CudaArray{T}))=(resize!(a, size(b)); copy!(a.arr, 1, b, 1, length(b)); a)
copy{A,T,N}(a::KUdense{A,T,N})=copy!(similar(a), a)
copy!{A,B,T}(a::KUdense{A,T}, ai::Integer, b::KUdense{B,T}, bi::Integer, n::Integer)=(copy!(a.arr, ai, b.arr, bi, n); a)

### EFFICIENT RESIZE

# Resize factor: 1.3 ensures a3 can be written where a0+a1 used to be
resizefactor(::Type{KUdense})=1.3

function resize!(a::KUdense, d::Dims)
    size(a)==d && return a
    n = prod(d)
    n > length(a.ptr) && resize!(a.ptr, round(Int,resizefactor(KUdense)*n+1))
    a.arr = arr(a.ptr, d)
    return a
end

resize!(a::KUdense, d::Int...)=resize!(a,d)

# Need to fix deepcopy so it does not create two arrays for arr and ptr:

cpucopy_internal{A<:Array}(x::KUdense{A},d::ObjectIdDict)=(haskey(d,x) ? d[x] : KUdense(copy(x.arr)))
cpucopy_internal{A<:CudaArray}(x::KUdense{A},d::ObjectIdDict)=(haskey(d,x) ? d[x] : KUdense(to_host(x.arr)))
gpucopy_internal{A<:Array}(x::KUdense{A},d::ObjectIdDict)=(haskey(d,x) ? d[x] : KUdense(CudaArray(x.arr)))
gpucopy_internal{A<:CudaArray}(x::KUdense{A},d::ObjectIdDict)=(haskey(d,x) ? d[x] : KUdense(copy(x.arr)))

randn!{A,T}(a::KUdense{A,T}, std=one(T), mean=zero(T))=(randn!(a.arr, std, mean); a)
rand!(a::KUdense)=(rand!(a.arr); a)
fill!{A,T}(a::KUdense{A,T},x)=(fill!(a.arr,convert(T,x)); a)

to_host{A<:CudaArray}(a::KUdense{A})=cpucopy(a)
issparse(a::KUdense)=false
