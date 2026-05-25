using LinearAlgebra
# exponential function with linear tail after x = x₀
function exp_lin(x; x₀=7)
    return x <= x₀ ? exp(x) : exp(x₀) + exp(x₀) * (x - x₀)
end

# inverse of exp_lin function
function exp_lin_inv(y; x₀=7)
    y₀ = exp(x₀)
    return y <= y₀ ? log(y) : x₀ + (y - y₀) / y₀
end

function simulateAR(T, ϕ, σₑ, μ=0)
    p = length(ϕ)
    x = zeros(2 * T)
    x[1:p] .= μ
    for t in (p+1):(2*T)
        x[t] = μ + ϕ ⋅ (reverse(x[t-p:t-1]) .- μ) + σₑ * randn()
    end
    return x[(T+1):end]
end

""" 
    interpParam2Obs(θ, groupSizes=nothing, dateVec=nothing; sample_t0=false, 
        output_t0=false, interpMethod=:constant) 

Interpolation from parameter time scale to observation time scale. 

- θ is Tgroup x p x nQuant, where Tgroup includes t=0 if sample_t0=true.
- sample_t0 = true if parameter at t=0 is sampled.
- output_t0 = true if the interpolated parameter at t=0 should be included in the output.
- groupSizes is a vector of length Tgroup - sample_t0, with the number of obs in each group.
"""
function interpParam2Obs(θ, groupSizes=nothing, dateVec=nothing; sample_t0=false,
    output_t0=false, interpMethod=:constant)

    Tgroup, p, nQuant = size(θ)
    if isnothing(groupSizes)
        groupSizes = ones(Int, Tgroup - sample_t0)
    end

    if sample_t0
        timeVec = 0:sum(groupSizes)
        groupSizes = [1; groupSizes]
    else
        timeVec = 1:sum(groupSizes)
    end

    # Set up date vectors
    if !isnothing(dateVec) && sample_t0
        timeUnit = dateVec[2] - dateVec[1]
        dateVec = [dateVec[1] - timeUnit; dateVec]
    end
    if isnothing(dateVec)
        dateVec = timeVec
    end
    if length(dateVec) != length(timeVec)
        error("Length of dateVec must match the time dimension of the data")
    end

    # Duplicate the group-level parameters to observation-level
    Tobs = sum(groupSizes)
    θobs = Array{eltype(θ)}(undef, Tobs, p, nQuant)
    idx = 1
    for t_group in 1:Tgroup
        for i in 1:groupSizes[t_group]
            for q in 1:nQuant
                if interpMethod == :constant
                    θobs[idx, :, q] .= θ[t_group, :, q]
                elseif interpMethod == :linear
                    if t_group == 1 || t_group == Tgroup
                        θobs[idx, :, q] .= θ[t_group, :, q]
                    else
                        weight = (i - 1) / groupSizes[t_group]
                        θobs[idx, :, q] .= (1 - weight) * θ[t_group, :, q] +
                                           weight * θ[t_group+1, :, q]
                    end
                else
                    error("Unsupported interpolation method: $interpMethod")
                end
            end
            idx += 1
        end
    end
    if sample_t0 && !output_t0
        θobs = θobs[2:end, :, :]
        dateVec = dateVec[2:end]
    end

    return θobs, dateVec
end

function scalingLabel(scaling)
    return scaling == :full ? "F" : (scaling == :diagonal ? "D" : "N")
end