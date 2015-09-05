# TODO: generalize to 3-D
# TODO: cpu implementation

type Pool5 <: Layer; dims; padding; stride; mode; pd; x; y; dx; dy; Pool5()=new(); end


function Pool5(dims::Dims;
               padding=ntuple(i->0, length(dims)),
               stride=dims,
               mode=CUDNN_POOLING_MAX)
    l=Pool5()
    l.dims=dims
    l.padding=padding; @assert length(padding)==length(dims)
    l.stride=stride;   @assert length(stride)==length(dims)
    l.mode=mode
    l.pd = PoolingDescriptor(dims, padding=padding, stride=stride, mode=mode)
    return l
end

Pool5(d::Int; o...)=Pool5((d,); o...)

function forw(l::Pool5, x; o...)
    initforw(l, x)
    kudnnPoolingForward(l.pd, l.x, l.y)
    return l.y
end

function initforw(l::Pool5, x)
    l.x = x
    pdims = ndims(x)-2
    if length(l.pd.dims) != pdims
        r = l.pd.dims[1]
        l.pd = PoolingDescriptor(ntuple(i->r, pdims))
    end
    similar!(l, :y, x, kudnnGetPoolingNdForwardOutputDim(l.pd, x))
end

function back(l::Pool5, dy; returndx=true, o...)
    returndx || return
    initback(l, dy)
    kudnnPoolingBackward(l.pd, l.y, l.dy, l.x, l.dx)
    return l.dx
end

function initback(l::Pool5, dy)
    @assert issimilar(dy, l.y)
    # l.dy = ((size(dy) == size(l.y)) ? dy : reshape(dy, size(l.y)))
    l.dy = dy
    similar!(l, :dx, l.x)
end


# Make things work with KUdense

KUDNN.kudnnGetPoolingNdForwardOutputDim(pd::PoolingDescriptor, x::KUdense)=kudnnGetPoolingNdForwardOutputDim(pd, x.arr)
KUDNN.kudnnPoolingForward(pd::PoolingDescriptor, x::KUdense, y::KUdense)=(kudnnPoolingForward(pd, x.arr, y.arr);y)
KUDNN.kudnnPoolingBackward(pd::PoolingDescriptor, y::KUdense, dy::KUdense, x::KUdense, dx::KUdense)=(kudnnPoolingBackward(pd, y.arr, dy.arr, x.arr, dx.arr);dx)


# Make things work with CPU (for now)

KUDNN.kudnnGetPoolingNdForwardOutputDim(pd::PoolingDescriptor, x::Array)=kudnnGetPoolingNdForwardOutputDim(pd, CudaArray(x))
KUDNN.kudnnPoolingForward(pd::PoolingDescriptor, x::Array, y::Array)=(y1=CudaArray(y);kudnnPoolingForward(pd, CudaArray(x), y1); copy!(y,1,y1,1,length(y)))
KUDNN.kudnnPoolingBackward(pd::PoolingDescriptor, y::Array, dy::Array, x::Array, dx::Array)=(dx1=CudaArray(dx);kudnnPoolingBackward(pd, CudaArray(y), CudaArray(dy), CudaArray(x), dx1); copy!(dx,1,dx1,1,length(dx)))


### DEAD CODE

# else

# warn("No cpu pool")

# end # if GPU

# Let these give error?
# Pool(x)=Pool()
# copy(l::Pool;o...)=Pool()
# forw(l::Pool,x;o...)=(l.x=l.y=x)
# back(l::Pool,dy;o...)=(l.dx=l.dy=dy)

# function forw(l::Pool, x; o...)
#     # error("CPU pool not implemented")
#     a = KUnet.Atype
#     KUnet.atype(CudaDynArray)
#     y = forw(copy(l), CudaDynArray(x); o...)
#     KUnet.atype(a)
#     l.x = x
#     l.y = to_host(y)
# end

# function back(l::Pool, dy; o...)
#     # error("CPU pool not implemented")
#     a = KUnet.Atype
#     KUnet.atype(CudaDynArray)
#     ll = copy(l); ll.y = CudaDynArray(l.y); ll.x = CudaDynArray(l.x)
#     dx = back(ll, CudaDynArray(dy); o...)
#     KUnet.atype(a)
#     l.dy = dy
#     l.dx = to_host(dx)
# end

