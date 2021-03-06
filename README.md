# StructArraysChainRules.jl

Defines [ChainRules](https://github.com/JuliaDiff/ChainRules.jl) adjoint rules for [StructArrays](https://github.com/JuliaArrays/StructArrays.jl).

## Examples

Try to run the following code (without loading this package first):

```julia
using Zygote, StructArrays
gradient(randn(2), randn(2)) do X,Y
    S = StructArray{Complex}((X,Y))
    sum(S).re + 2sum(S).im
end
```

You will get an error.
Now load this package and run that again.


## Related

See also: https://github.com/cossio/ZygoteStructArrays.jl.
StructArraysChainRules should replace that eventually.