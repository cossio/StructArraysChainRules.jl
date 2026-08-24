import ChainRulesCore
using ChainRulesCore: AbstractZero, NoTangent, ZeroTangent, Tangent, unthunk

_componentnames(S::StructArrays.StructArray) = propertynames(StructArrays.components(S))

# Extract the tangent of one field from the tangent of a struct-like element.
_elfield(x, name::Symbol) = getproperty(x, name)
_elfield(x::Union{Tangent,NamedTuple}, name::Symbol) = _zero2zero(getproperty(x, name))
_elfield(x::AbstractZero, ::Symbol) = x
_elfield(::Nothing, ::Symbol) = ZeroTangent()

_zero2zero(x) = x
_zero2zero(::Nothing) = ZeroTangent()

# Normalize a tangent `Δ` for a StructArray with component names `names` into a
# NamedTuple mapping each component name to its array (or zero) tangent.
function _component_tangents(Δ, names::NTuple{N,Symbol}) where {N}
    Δ = unthunk(Δ)
    return _component_tangents_norm(Δ, names)
end

_component_tangents_norm(Δ::StructArrays.StructArray, names) =
    NamedTuple{names}(map(name -> getproperty(StructArrays.components(Δ), name), names))
function _component_tangents_norm(Δ::Union{Tangent,NamedTuple}, names)
    Δc = _zero2zero(hasproperty(Δ, :components) ? getproperty(Δ, :components) : Δ)
    return NamedTuple{names}(map(name -> _elfield(Δc, name), names))
end
_component_tangents_norm(Δ::AbstractArray, names) =
    NamedTuple{names}(map(name -> map(x -> _elfield(x, name), Δ), names))

# Materialize a possibly-zero component tangent as an array matching component `c`.
_materialize(Δ, c) = Δ
_materialize(::Union{AbstractZero,Nothing}, c) = zero(c)

# Rebuild an elementwise tangent array (tangent of an array of structs) from
# per-component tangents.
function _element_tangents(S::StructArrays.StructArray, Δc::NamedTuple{names}) where {names}
    comps = StructArrays.components(S)
    filled = NamedTuple{names}(map(name -> _materialize(getfield(Δc, name), getproperty(comps, name)), names))
    if eltype(S) <: Complex && names === (:re, :im)
        return complex.(filled.re, filled.im)
    end
    return StructArrays.StructArray(filled)
end

# Map canonical per-component tangents back onto the constructor argument layout.
_arg_tangent(c::Tuple, Δc::NamedTuple) = Tangent{typeof(c)}(values(Δc)...)
_arg_tangent(c::NamedTuple{K}, Δc::NamedTuple) where {K} =
    Tangent{typeof(c)}(; NamedTuple{K}(map(k -> getfield(Δc, k), K))...)

# Construction from a Tuple or NamedTuple of component arrays.
function ChainRulesCore.rrule(::Type{SA}, c::Union{Tuple,NamedTuple}) where {SA<:StructArrays.StructArray}
    S = SA(c)
    names = _componentnames(S)
    function StructArray_pullback(Δ)
        ΔS = unthunk(Δ)
        ΔS isa AbstractZero && return (NoTangent(), ΔS)
        Δc = _component_tangents(ΔS, names)
        return (NoTangent(), _arg_tangent(c, Δc))
    end
    return S, StructArray_pullback
end

# Construction from an array of structs.
function ChainRulesCore.rrule(::Type{SA}, v::AbstractArray) where {SA<:StructArrays.StructArray}
    S = SA(v)
    names = _componentnames(S)
    function StructArray_from_array_pullback(Δ)
        ΔS = unthunk(Δ)
        ΔS isa AbstractZero && return (NoTangent(), ΔS)
        Δc = _component_tangents(ΔS, names)
        return (NoTangent(), _element_tangents(S, Δc))
    end
    return S, StructArray_from_array_pullback
end

# Scalar indexing, keeping the tangent structural so that untouched components
# propagate exact (hard) zeros.
function ChainRulesCore.rrule(::typeof(getindex), S::StructArrays.StructArray, I::Integer...)
    el = S[I...]
    names = _componentnames(S)
    comps = StructArrays.components(S)
    function getindex_pullback(Δ)
        Δel = unthunk(Δ)
        Δel isa AbstractZero && return (NoTangent(), Δel, map(_ -> NoTangent(), I)...)
        parts = NamedTuple{names}(map(names) do name
            δ = _elfield(Δel, name)
            if δ isa Union{AbstractZero,Nothing} || (δ isa Number && iszero(δ))
                ZeroTangent()
            else
                a = zero(getproperty(comps, name))
                a[I...] = δ
                a
            end
        end)
        ΔS = Tangent{typeof(S)}(; components=Tangent{typeof(comps)}(; parts...))
        return (NoTangent(), ΔS, map(_ -> NoTangent(), I)...)
    end
    return el, getindex_pullback
end
