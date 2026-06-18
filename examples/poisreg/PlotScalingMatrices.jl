Svec_median = median(Svec, dims=4)[:, :, :, 1]
plotScaling = plot(xlabel="time, " * L"t", title="Scaling matrix elements over time", lw=2)
count = 0
for i in 1:p
    for j in 1:i
        count = count + 1
        style = isodd(count) ? :solid : :dash
        plot!(plotScaling, Svec_median[i, j, :], label="Svec[$i, $j]", xlabel="time, " * L"t", title="Scaling matrix element Svec[$i, $j]", lw=2, color=colors[count],
            linestyle=style)
    end
end
plotScaling

eigScaleMat = zeros(T, p)
count = 1
for t in 1:size(Svec_median, 3)
    eigScaleMat[count:(count+groupSizes[t]-1), :] .= repeat(eigvals(Svec_median[:, :, t])',
        groupSizes[t])
    count = count + groupSizes[t]
end
plot(eigScaleMat, ylim=(0, maximum(eigScaleMat)), xlabel="time, " * L"t",
    title="Eigenvalues of the scaling matrix over time", lw=2)