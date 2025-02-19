abstract type PolicyGradient end

struct Static <: PolicyGradient end

# Vanilla policy gradient
struct VPG{T<:AbstractFloat} <: PolicyGradient
    η::T
end

function learning_step!(parameters::AbstractArray, gd::GradientData, opt::VPG)
    parameters .= parameters + opt.η * gd.∇j
end

# Baseline policy gradient
struct BLPG{T<:AbstractFloat} <: PolicyGradient
    η::T
end

function learning_step!(parameters::AbstractArray, gd::GradientData, opt::BLPG)
    parameters .= parameters + opt.η * (gd.∇j - gd.j * gd.∇logq_forward)
end

# Baseline adaptive policy gradient
struct BLAPG{T<:AbstractFloat} <: PolicyGradient
    δ::T
    ϵid::T
end

BLAPG(δ::T) where {T<:AbstractFloat} = BLAPG(δ, zero(T))

function learning_step!(parameters::AbstractArray, gd::GradientData, opt::BLAPG)
    η = sqrt(2opt.δ / (dot(gd.∇j, gd.∇j) + opt.ϵid))
    parameters .= parameters + η * (gd.∇j - gd.j * gd.∇logq_forward)
end

# Natural policy gradient
struct NPG{T<:AbstractFloat} <: PolicyGradient
    η::T
    ϵid::T
end

NPG(η::T) where {T<:AbstractFloat} = NPG(η, zero(T))

function learning_step!(parameters::AbstractArray, gd::GradientData, opt::NPG)
    parameters .= parameters + opt.η * inv(gd.g + opt.ϵid * I) * gd.∇j
end

# Adaptive natural policy gradient
struct ANPG{T<:AbstractFloat} <: PolicyGradient
    δ::T
    ϵid::T
end

ANPG(δ::T) where {T<:AbstractFloat} = ANPG(δ, zero(T))

function learning_step!(parameters::AbstractArray, gd::GradientData, opt::ANPG)
    F⁻¹ = inv(gd.g + opt.ϵid * I)
    η = sqrt(2opt.δ / (gd.∇j' * (F⁻¹ * gd.∇j)))
    parameters .= parameters + η * F⁻¹ * gd.∇j
end

# Baseline adaptive natural policy gradient
struct BLANPG{T<:AbstractFloat} <: PolicyGradient
    δ::T
    ϵid::T
end

BLANPG(δ::T) where {T<:AbstractFloat} = BLANPG(δ, zero(T))

function learning_step!(parameters::AbstractArray, gd::GradientData, opt::BLANPG)
    F⁻¹ = inv(gd.g + opt.ϵid * I)
    ∇̄j = gd.∇j - gd.j * gd.∇logq_forward
    η = sqrt(2opt.δ / (∇̄j' * (F⁻¹ * ∇̄j)))
    parameters .= parameters + η * F⁻¹ * ∇̄j
end

nothing
