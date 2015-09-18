module NN

typealias Net Array{Layer, 1}

type Layer
    w  # each row is the weights of a neuron 
    b  # bias
    f  # activation
    z  # 
    o  # output
    fx # preprocess function
end

@doc """ x: nxm
""" ->
function forward(l::layer, x)
    l.x = l.fx(l, x) # may be changed to mult-patch
    l.z = l.w*l.x' + l.b
    l.o = l.f(l, z)
end

forward(n::Net, x) = (for l=n; x=forward(l,x) end; x)


@doc """ y is the predicted
        dy is the ground truth
""" ->
function softmaxloss(y, dy)

    




end
