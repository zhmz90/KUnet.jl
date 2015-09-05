# TODO: generalize to N-D
# TODO: cpu implementation

using KUDNN

type Conv5 <: Layer; w; x; y; dx; dy; Conv5(p::KUparam)=new(p); end

Conv5(d...; o...)=Conv5(KUparam(d...; o...))

Conv5(nout::Integer, width::Integer; o...)=Conv5(KUparam(width, 0, nout; o...))

param(l::Conv5)=l.w

function forw(l::Conv5, x; o...)
    initforw(l,x)
    kudnnConvolutionForward(l.x, l.w, l.y)
    return l.y
end

function initforw(l::Conv5, x)
    n = ndims(x)
    c = size(x)[n-1]  # x dims are (x1, x2, ..., channels, images)
    if isempty(l.w) 
        nz(l.w,:init,nothing) || (l.w.init = initxavier)
        r = size(l.w, 1)
        o = size(l.w, ndims(l.w))
        wsize = ntuple(i->(i<n-1 ? r : i==n-1 ? c : o), n)
        init(l.w, eltype(x), wsize)
    end
    @assert eltype(x) == eltype(l.w) "$(eltype(x)) != $(eltype(l.w))"
    @assert n == ndims(l.w)
    @assert c == size(l.w)[end-1]
    similar!(l, :y, x, kudnnGetConvolutionNdForwardOutputDim(x, l.w))
    l.x = x
end

function back(l::Conv5, dy; returndx=true, o...)
    initback(l, dy, returndx)
    kudnnConvolutionBackwardFilter(l.x, l.dy, l.w)
    returndx && kudnnConvolutionBackwardData(l.w, l.dy, l.dx)
end

function initback(l::Conv5, dy, returndx)
    @assert issimilar(dy, l.y)
    # @assert eltype(dy) == eltype(l.y)
    # l.dy = (size(dy) == size(l.y) ? dy : reshape(dy, size(l.y)))
    l.dy = dy
    initdiff(l.w)
    returndx && similar!(l, :dx, l.x)
end

# Make things work with KUdense

KUDNN.kudnnConvolutionForward(x::KUdense, w::KUparam, y::KUdense)=(kudnnConvolutionForward(x.arr, w.arr, y.arr);y)
KUDNN.kudnnConvolutionBackwardFilter(x::KUdense, dy::KUdense, w::KUparam)=(kudnnConvolutionBackwardFilter(x.arr, dy.arr, w.diff);w)
KUDNN.kudnnConvolutionBackwardData(w::KUparam, dy::KUdense, dx::KUdense)=(kudnnConvolutionBackwardData(w.arr, dy.arr, dx.arr);dx)

# Make things work with CPU (for now)

KUDNN.kudnnConvolutionForward(x::Array, w::Array, y::Array)=(y1=CudaArray(y);kudnnConvolutionForward(CudaArray(x), CudaArray(w), y1);copy!(y,1,y1,1,length(y)))
KUDNN.kudnnConvolutionBackwardFilter(x::Array, dy::Array, w::Array)=(w1=CudaArray(w);kudnnConvolutionBackwardFilter(CudaArray(x), CudaArray(dy), w1); copy!(w,1,w1,1,length(w)))
KUDNN.kudnnConvolutionBackwardData(w::Array, dy::Array, dx::Array)=(dx1=CudaArray(dx);kudnnConvolutionBackwardData(CudaArray(w), CudaArray(dy), dx1); copy!(dx,1,dx1,1,length(dx)))

