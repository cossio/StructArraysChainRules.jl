using Test, Random, Zygote, StructArrays, StructArraysChainRules

@test gradient(randn(2), randn(2)) do X,Y
    S = StructArray{Complex}((X,Y))
    sum(S.re) + 2sum(S.im)
end == ([1.0, 1.0], [2.0, 2.0])

@test gradient(randn(2), randn(2)) do X,Y
    S = StructArray{Complex}((im = Y, re = X))
    sum(S.re) + 2sum(S.im)
end == ([1.0, 1.0], [2.0, 2.0])

@test gradient(randn(2), randn(2)) do X,Y
    S = StructArray{Complex}((re = X, im = Y))
    sum(S.re) + 2sum(S.im)
end == ([1.0, 1.0], [2.0, 2.0])

@test gradient(randn(2), randn(2)) do X,Y
    S = StructArray{Complex}((X,Y))
    sum(S).re + 2sum(S).im
end == ([1.0, 1.0], [2.0, 2.0])

@test gradient(randn(2), randn(2)) do X,Y
    S = StructArray{Complex}((re = X, im = Y))
    sum(S).re + 2sum(S).im
end == ([1.0, 1.0], [2.0, 2.0])

@test gradient(randn(2), randn(2)) do X,Y
    S = StructArray{Complex}((im = Y, re = X))
    sum(S).re + 2sum(S).im
end == ([1.0, 1.0], [2.0, 2.0])

@test gradient(randn(2), randn(2)) do X, Y
    S = StructArray{Complex}((re = X, im = Y))
    S[1].re
end == ([1.0, 0.0], nothing)

@test gradient(randn(2), randn(2)) do X, Y
    S = StructArray{Complex}((re = X, im = Y))
    S[1].re + S[1].im
end == ([1.0, 0.0], [1.0, 0.0])

@test gradient(randn(2), randn(2)) do X, Y
    S = StructArray{Complex}((re = X, im = Y))
    S[1].re + S[2].re
end == ([1.0, 1.0], nothing)


struct Point
    x::Float64; y::Float64
end

@test gradient(randn(2), randn(2)) do X, Y
    S = StructArray(Point.(X, Y))
    sum(S.x) + sum(S.y)
end == ([1.0, 1.0], [1.0, 1.0])

@test gradient(randn(2), randn(2)) do X, Y
    S = StructArray{Point}((x = X, y = Y))
    fun(p::Point) = p.x + 2p.y
    sum(fun.(S))
end == ([1.0, 1.0], [2.0, 2.0])

#1
@test gradient(randn(2), randn(2)) do X, Y
    S = StructArray(Complex.(X, Y))
    sum(S.re) + 2sum(S.im)
end == ([1.0, 1.0], [2.0, 2.0])

@test gradient(randn(2), randn(2)) do X, Y
    S = StructArray{Complex}((re = X, im = Y))
    fun(z::Complex) = real(z) + 2imag(z)
    sum(fun.(S))
end == ([1.0, 1.0], [2.0, 2.0])
