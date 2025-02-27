"""
    abstract type PolicyGradient end

Abstract type representing policy gradient optimisers.

Concrete subtypes must implement:
- `learning_step!(parameters, gd, opt)`: Update the parameters based on the gradient information
"""
abstract type PolicyGradient end

"""
    Static <: PolicyGradient end

Static policy gradient optimiser that does not update the parameters.
"""
struct Static <: PolicyGradient end

"""
    VPG{T<:AbstractFloat} <: PolicyGradient

Vanilla policy gradient optimiser.
"""
struct VPG{T<:AbstractFloat} <: PolicyGradient
    η::T
end

"""
    learning_step!(parameters::AbstractArray, gd::GradientData, opt::VPG)

Update the parameters based on the gradient information.
"""
function learning_step!(parameters::AbstractArray, gd::GradientData, opt::VPG)
    parameters .= parameters + opt.η * gd.∇j
end

"""
    BLPG{T<:AbstractFloat} <: PolicyGradient

Baseline policy gradient optimiser.
"""
struct BLPG{T<:AbstractFloat} <: PolicyGradient
    η::T
end

"""
    learning_step!(parameters::AbstractArray, gd::GradientData, opt::BLPG)

Update the parameters based on the gradient information.
"""
function learning_step!(parameters::AbstractArray, gd::GradientData, opt::BLPG)
    parameters .= parameters + opt.η * (gd.∇j - gd.j * gd.∇logq_forward)
end

"""
    BLAPG{T<:AbstractFloat} <: PolicyGradient

Baseline Adaptive Policy Gradient (BLAPG) optimiser.
"""
struct BLAPG{T<:AbstractFloat} <: PolicyGradient
    δ::T
    ϵid::T
end

"""
    BLAPG(δ::T) where {T<:AbstractFloat}

Construct a BLAPG optimiser with default `ϵid=0`.
"""
BLAPG(δ::T) where {T<:AbstractFloat} = BLAPG(δ, zero(T))

"""
    learning_step!(parameters::AbstractArray, gd::GradientData, opt::BLAPG)

Update the parameters based on the gradient information.
"""
function learning_step!(parameters::AbstractArray, gd::GradientData, opt::BLAPG)
    η = sqrt(2opt.δ / (dot(gd.∇j, gd.∇j) + opt.ϵid))
    parameters .= parameters + η * (gd.∇j - gd.j * gd.∇logq_forward)
end

"""
    NPG{T<:AbstractFloat} <: PolicyGradient

Natural policy gradient optimiser.
"""
struct NPG{T<:AbstractFloat} <: PolicyGradient
    η::T
    ϵid::T
end

"""
    NPG(η::T) where {T<:AbstractFloat}

Construct a NPG optimiser with default `ϵid=0`.
"""
NPG(η::T) where {T<:AbstractFloat} = NPG(η, zero(T))

"""
    learning_step!(parameters::AbstractArray, gd::GradientData, opt::NPG)

Update the parameters based on the gradient information.
"""
function learning_step!(parameters::AbstractArray, gd::GradientData, opt::NPG)
    parameters .= parameters + opt.η * inv(gd.g + opt.ϵid * I) * gd.∇j
end


"""
    ANPG{T<:AbstractFloat} <: PolicyGradient

Adaptive natural policy gradient optimiser.
"""
struct ANPG{T<:AbstractFloat} <: PolicyGradient
    δ::T
    ϵid::T
end

"""
    ANPG(δ::T) where {T<:AbstractFloat}

Construct an ANPG optimiser with default `ϵid=0`.
"""
ANPG(δ::T) where {T<:AbstractFloat} = ANPG(δ, zero(T))

"""
    learning_step!(parameters::AbstractArray, gd::GradientData, opt::ANPG)

Update the parameters based on the gradient information.
"""
function learning_step!(parameters::AbstractArray, gd::GradientData, opt::ANPG)
    F⁻¹ = inv(gd.g + opt.ϵid * I)
    η = sqrt(2opt.δ / (gd.∇j' * (F⁻¹ * gd.∇j)))
    parameters .= parameters + η * F⁻¹ * gd.∇j
end


"""
    BLANPG{T<:AbstractFloat} <: PolicyGradient

Baseline adaptive natural policy gradient optimiser.
"""
struct BLANPG{T<:AbstractFloat} <: PolicyGradient
    δ::T
    ϵid::T
end

"""
    BLANPG(δ::T) where {T<:AbstractFloat}

Construct a BLANPG optimiser with default `ϵid=0`.
"""
BLANPG(δ::T) where {T<:AbstractFloat} = BLANPG(δ, zero(T))

"""
    learning_step!(parameters::AbstractArray, gd::GradientData, opt::BLANPG)

Update the parameters based on the gradient information.
"""
function learning_step!(parameters::AbstractArray, gd::GradientData, opt::BLANPG)
    F⁻¹ = inv(gd.g + opt.ϵid * I)
    ∇̄j = gd.∇j - gd.j * gd.∇logq_forward
    η = sqrt(2opt.δ / (∇̄j' * (F⁻¹ * ∇̄j)))
    parameters .= parameters + η * F⁻¹ * ∇̄j
end

nothing
