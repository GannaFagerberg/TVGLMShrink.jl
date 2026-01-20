# exponential function with linear tail after x = x₀
function exp_lin(x; x₀ = 7)
    return x <= x₀ ? exp(x) : exp(x₀) + exp(x₀)*(x - x₀)
end

# inverse of exp_lin function
function exp_lin_inv(y; x₀ = 7)
    y₀ = exp(x₀)
    return y <= y₀ ? log(y) : x₀ + (y - y₀)/y₀
end


