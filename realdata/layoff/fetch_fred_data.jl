"""
Fetch and merge FRED series

Response:    LNS13023654  — job losers on layoff as % of total unemployed
Alt response LNS13023622  - job losers + other unemployed on layoff as % of total unemployed
Covariates:
  Strong:    ICSA          — initial jobless claims (monthly avg of weekly)
             BAA-GS10      — Moody's Baa minus 10yr Treasury credit spread
             VIXCLS        — VIX (monthly avg)
  Weak/shrinkage targets:
             UMCSENT       — University of Michigan consumer sentiment
             DCOILWTICO    — WTI crude oil price (monthly avg)
  Extra:     CPIAUCSL_pc1  — CPI inflation YoY %

All series aggregated to monthly frequency. Dataset trimmed to the
common non-missing window across all series.

Usage:  julia fetch_fred_data.jl
"""

using HTTP, JSON3, DataFrames, CSV, Dates, Statistics, Printf, Plots
using Utils: mvcolors as colors

const API_KEY = "b22f5fe3e6514e1a8e775befd63a9cf4"
const BASE = "https://api.stlouisfed.org/fred/series/observations"

figFolder = joinpath(@__DIR__, "figs")
dataFolder = joinpath(@__DIR__, "data")

# ── fetch one FRED series ─────────────────────────────────────────────────────

function fetch_fred(series_id; freq="m", agg="avg", units=nothing)
    params = Dict(
        "series_id" => series_id,
        "api_key" => API_KEY,
        "file_type" => "json",
        "frequency" => freq,
        "aggregation_method" => agg,
    )
    isnothing(units) || (params["units"] = units)

    query = join(["$k=$v" for (k, v) in params], "&")
    url = "$BASE?$query&limit=100000"

    resp = HTTP.get(url; connect_timeout=30, read_idle_timeout=30)
    data = JSON3.read(String(resp.body))
    obs = data["observations"]

    dates = Date.(getindex.(obs, "date"))
    values = map(o -> o["value"] == "." ? missing : tryparse(Float64, o["value"]), obs)

    df = DataFrame(:date => dates, Symbol(series_id) => values)
    nnan = count(ismissing, df[!, 2])
    println("  $series_id: $(nrow(df)) obs, $(first(dates)) – $(last(dates)), missing: $nnan")
    return df
end

# ── fetch all series ──────────────────────────────────────────────────────────

println("Fetching from FRED API...")

dfs = [
    fetch_fred("LNS13023654"),              # layoff share — response
    fetch_fred("LNS13023622"),              # lob losers (layoff + other) as % of unemployed
    fetch_fred("CPIAUCSL"; units="pc1"),    # CPI YoY inflation
    fetch_fred("ICSA"),                     # initial claims
    fetch_fred("BAA"),                      # Baa corporate yield
    fetch_fred("GS10"),                     # 10yr Treasury yield
    fetch_fred("VIXCLS"),                   # VIX
    fetch_fred("UMCSENT"),                  # consumer sentiment
    fetch_fred("DCOILWTICO"),               # WTI oil price
]

rename!(dfs[3], :CPIAUCSL => :CPIAUCSL_pc1)

# ── outer merge on date ───────────────────────────────────────────────────────

df = foldl((a, b) -> outerjoin(a, b, on=:date), dfs)
sort!(df, :date)
df[!, :credit_spread] = df[!, :BAA] .- df[!, :GS10]

# ── first and last non-missing date per series ────────────────────────────────

println("\nFirst / last non-missing date per series:")
for col in names(df)
    col == "date" && continue
    idx_first = findfirst(!ismissing, df[!, col])
    idx_last = findlast(!ismissing, df[!, col])
    if isnothing(idx_first) || isnothing(idx_last)
        println("  $col: all missing")
    else
        println("  $col: $(df.date[idx_first]) to $(df.date[idx_last])")
    end
end

# ── trim ragged ends, keep all interior rows ──────────────────────────────────

first_complete = df.date[findfirst(completecases(df))]
last_complete = df.date[findlast(completecases(df))]
df_out = df[first_complete.<=df.date.<=last_complete, :]

println("\nOuter merge: $(nrow(df)) rows")
println("Trimmed window: $first_complete – $last_complete  ($(nrow(df_out)) rows)")
n_incomplete = nrow(df_out) - sum(completecases(df_out))
println("Interior incomplete rows retained: $n_incomplete")

# ── log-transform skewed covariates ──────────────────────────────────────────

df_out[!, :log_ICSA] = passmissing(log).(df_out.ICSA)
df_out[!, :log_DCOILWTICO] = passmissing(log).(df_out.DCOILWTICO)

# ── interpolate isolated single missing values ────────────────────────────────

println("\nInterpolating isolated missing values:")
n_interp = 0
for col in names(df_out)
    col in ("date", "t") && continue
    for i in 2:(nrow(df_out)-1)
        if ismissing(df_out[i, col]) &&
           !ismissing(df_out[i-1, col]) &&
           !ismissing(df_out[i+1, col])
            df_out[i, col] = (df_out[i-1, col] + df_out[i+1, col]) / 2
            println("  $col at $(df_out.date[i])")
            n_interp += 1
        end
    end
end
n_interp == 0 && println("  none needed")

# ── flag runs of consecutive missing values (length > 1) ─────────────────────

println("\nConsecutive missing runs (length > 1) per column:")
found_any = false
for col in names(df_out)
    col in ("date", "t") && continue
    run_len = 0
    for i in 1:nrow(df_out)
        if ismissing(df_out[i, col])
            run_len += 1
        else
            if run_len > 1
                println("  $col: $run_len consecutive missing ending at $(df_out.date[i-1])")
                found_any = true
            end
            run_len = 0
        end
    end
    run_len > 1 && println("  $col: $run_len consecutive missing at end of series")
end
found_any || println("  none found")

# ── convert layoff share from percent to proportion (0–1) ────────────────────

df_out[!, :y] = df_out.LNS13023654 ./ 100.0
df_out[!, :yalt] = df_out.LNS13023622 ./ 100.0


# ── Smithson-Verkuilen squeeze to keep y strictly in (0,1) ───────────────────

N = nrow(df_out)
n_boundary = count(x -> x == 0.0 || x == 1.0, df_out.y)
println("\nSmithson-Verkuilen: $n_boundary boundary observations squeezed " *
        "($(round(100*n_boundary/N, digits=2))%)")
df_out[!, :y] = (df_out.y .* (N - 1) .+ 0.5) ./ N
df_out[!, :yalt] = (df_out.yalt .* (N - 1) .+ 0.5) ./ N


# ── add time index ────────────────────────────────────────────────────────────

df_out[!, :t] = 1:nrow(df_out)

# ── variable summary ──────────────────────────────────────────────────────────

println("\n── Variable summary ──────────────────────────────────────────")
for col in [:y, :yalt, :CPIAUCSL_pc1, :log_ICSA, :credit_spread, :VIXCLS, :UMCSENT,
    :log_DCOILWTICO]
    v = skipmissing(df_out[!, col])
    @printf("  %-20s  min=%7.3f  mean=%7.3f  max=%7.3f  sd=%6.3f\n",
        col, minimum(v), mean(v), maximum(v), std(v))
end

println("\nMissing values per column:")
any_missing = false
for col in names(df_out)
    n = count(ismissing, df_out[!, col])
    if n > 0
        println("  $col: $n missing")
        any_missing = true
    end
end
any_missing || println("  none")

# ── rename columns ────────────────────────────────────────────────────────────

rename!(df_out, Dict(
    :y => :layoff_share,
    :yalt => :joblosers_share,
    :CPIAUCSL_pc1 => :cpi_infl,
    :log_ICSA => :log_claims,
    :ICSA => :claims,
    :credit_spread => :credit_spr,
    :VIXCLS => :vix,
    :UMCSENT => :sentiment,
    :log_DCOILWTICO => :log_oil,
    :DCOILWTICO => :oil,
))

# ── add lags ──────────────────────────────────────────────────────────────────

nLags = 4
lag_cols = [:layoff_share, :joblosers_share, :log_claims, :credit_spr, :vix, :cpi_infl, :sentiment, :log_oil]

for lag in 1:nLags
    for col in lag_cols
        lagname = Symbol("$(col)_lag$(lag)")
        df_out[!, lagname] = [fill(missing, lag); df_out[1:end-lag, col]]
    end
end

lag_col_names = [Symbol("$(col)_lag$(lag)") for lag in 1:nLags for col in lag_cols]

# ── save ──────────────────────────────────────────────────────────────────────

out_cols = [
    :t, :date, :layoff_share, :joblosers_share,
    :cpi_infl, :log_claims, :claims,
    :credit_spr, :vix, :sentiment,
    :log_oil, :oil,
    lag_col_names...
]

df_out = df_out[:, out_cols]  # keep only the columns we care about

CSV.write(joinpath(dataFolder, "layoff_beta_data.csv"), df_out)
println("\nSaved layoff_beta_data.csv  ($(nrow(df_out)) rows × $(length(out_cols)) cols)")


## PLOTS

# choice of response variable for plots
response = :joblosers_share

function lag_scatter(df, response, covariate, period_name; lags=0:4, start_date=nothing, end_date=nothing)
    sub = df
    !isnothing(start_date) && (sub = sub[sub.date.>=start_date, :])
    !isnothing(end_date) && (sub = sub[sub.date.<=end_date, :])

    n_lags = length(lags)
    ncols = min(n_lags, 3)
    nrows = ceil(Int, n_lags / ncols)

    plots = []
    for lag in lags
        y_vec = lag == 0 ? sub[!, response] : sub[!, response][1:end-lag]
        x_vec = lag == 0 ? sub[!, covariate] : sub[!, covariate][1:end-lag]
        valid = .!ismissing.(x_vec) .& .!ismissing.(y_vec)
        r = cor(Float64.(x_vec[valid]), Float64.(y_vec[valid]))
        p = scatter(x_vec, y_vec;
            xlabel="$covariate (lag $lag)",
            ylabel=string(response),
            title="lag $lag  (r=$(round(r, digits=2)))",
            markersize=3,
            markerstrokewidth=0,
            alpha=0.4,
            legend=false)
        push!(plots, p)
    end

    plt = plot(plots...,
        layout=n_lags,
        size=(300 * ncols, 280 * nrows),
        plot_title="$covariate — $period_name")
    savefig(joinpath(figFolder, "scatter_$(covariate)_$(period_name).png"))
    println("Saved scatter_$(covariate)_$(period_name).png")
    return plt
end

# ── covariates to plot ────────────────────────────────────────────────────────
covariates = [:log_claims, :credit_spr, :vix, :cpi_infl, :sentiment, :log_oil]

# ── period definitions ────────────────────────────────────────────────────────
periods = [
    ("pre-covid", Date("1990-01-01"), Date("2020-01-01")),
    ("covid", Date("2020-02-01"), Date("2022-06-01")),
    ("post-covid", Date("2022-07-01"), nothing),
]

# ── call for all combinations ─────────────────────────────────────────────────
for cov in covariates, (name, t_start, t_end) in periods
    lag_scatter(df_out, response, cov, name; start_date=t_start, end_date=t_end)
end

# plot time series of layoff/job_losers and color-code the three periods. just time series, no scatterplots.
function plot_response_time_series(df, response, periods)
    period_colors = [colors[1], colors[3], colors[2]]
    p = plot(
        xlabel="Date",
        ylabel="Layoff Share",
        title="Layoff Share", ylim=(0, 1),
        legend=:topleft, grid=false)
    for (k, (name, t_start, t_end)) in enumerate(periods)
        in_period = isnothing(t_end) ? (df.date .>= t_start) : ((df.date .>= t_start) .& (df.date .<= t_end))
        sub = df[in_period, [:date, response]]
        valid = .!ismissing.(sub[!, response])
        plot!(p, sub.date[valid], sub[!, response][valid];
            color=period_colors[k],
            linewidth=2,
            label=string(name))
    end
    return p
end

# plot the layoff share time series with the three periods color-coded
plot_response_time_series(df_out, response, periods)

savefig(joinpath(figFolder, "layoff_time_series.svg"))


# plot time series of layoff and color-code the three periods. just time series, no scatterplots.
function plot_vix_time_series(df, periods)
    period_colors = [colors[1], colors[3], colors[2]]
    p = plot(
        xlabel="Date",
        ylabel="VIX lag1",
        title="VIX",
        legend=:topleft, grid=false)
    for (k, (name, t_start, t_end)) in enumerate(periods)
        in_period = isnothing(t_end) ? (df.date .>= t_start) : ((df.date .>= t_start) .& (df.date .<= t_end))
        sub = df[in_period, [:date, :vix_lag1]]
        valid = .!ismissing.(sub.vix_lag1)
        plot!(p, sub.date[valid], sub.vix_lag1[valid];
            color=period_colors[k],
            linewidth=2,
            label=string(name))
    end
    return p
end

# use the function above to plot the layoff share time series with the three periods color-coded
plot_vix_time_series(df_out, periods)

savefig(joinpath(figFolder, "vixlag1_time_series.svg"))

# scatterplot of the covariates against each other with three colors for the three periods
function covariate_scatter_matrix(df, vars, periods)
    n_cov = length(vars)
    period_colors = [colors[1], colors[3], colors[2]]
    plots = []
    for i in 1:n_cov, j in 1:n_cov
        if i == j
            p = plot(
                xlabel=string(vars[i]),
                ylabel="Frequency",
                title=string(vars[i]),
                legend=(i == 1 && j == 1))
            for (k, (name, t_start, t_end)) in enumerate(periods)
                in_period = isnothing(t_end) ? (df.date .>= t_start) : ((df.date .>= t_start) .& (df.date .<= t_end))
                x_all = df[!, vars[i]]
                valid = in_period .& .!ismissing.(x_all)
                if any(valid)
                    histogram!(p, x_all[valid];
                        bins=25,
                        alpha=0.5,
                        normalize=true,
                        color=period_colors[k],
                        label=string(name))
                end
            end
        else
            p = plot(
                xlabel=string(vars[j]),
                ylabel=string(vars[i]),
                title="",
                legend=(i == 1 && j == 1))
            x_all = df[!, vars[j]]
            y_all = df[!, vars[i]]
            for (k, (name, t_start, t_end)) in enumerate(periods)
                in_period = isnothing(t_end) ? (df.date .>= t_start) : ((df.date .>= t_start) .& (df.date .<= t_end))
                valid = in_period .& .!ismissing.(x_all) .& .!ismissing.(y_all)
                if any(valid)
                    scatter!(p, x_all[valid], y_all[valid];
                        markersize=3,
                        markerstrokewidth=0,
                        alpha=1,
                        color=period_colors[k],
                        label=string(name))
                end
            end
        end
        push!(plots, p)
    end
    plt = plot(plots...,
        layout=(n_cov, n_cov),
        size=(300 * n_cov, 300 * n_cov),
        plot_title="Covariate Scatter Matrix")
    savefig(joinpath(figFolder, "covariate_scatter_matrix.png"))
    println("Saved covariate_scatter_matrix.png")
    return plt
end

# one scatterplot matrix per period, using the same period colors
function covariate_scatter_matrix_multiple(df, vars, periods)
    n_cov = length(vars)
    period_colors = [colors[1], colors[3], colors[2]]
    out_plots = Dict{String,Any}()

    for (k, (name, t_start, t_end)) in enumerate(periods)
        in_period = isnothing(t_end) ? (df.date .>= t_start) : ((df.date .>= t_start) .& (df.date .<= t_end))
        sub = df[in_period, :]
        period_color = period_colors[k]

        plots = []
        for i in 1:n_cov, j in 1:n_cov
            if i == j
                x_all = sub[!, vars[i]]
                valid = .!ismissing.(x_all)
                p = histogram(x_all[valid];
                    bins=25,
                    alpha=0.5,
                    normalize=true,
                    color=period_color,
                    xlabel=string(vars[i]),
                    ylabel="Frequency",
                    title=string(vars[i]),
                    legend=false)
            else
                x_all = sub[!, vars[j]]
                y_all = sub[!, vars[i]]
                valid = .!ismissing.(x_all) .& .!ismissing.(y_all)
                p = scatter(x_all[valid], y_all[valid];
                    markersize=3,
                    markerstrokewidth=0,
                    alpha=1,
                    color=period_color,
                    xlabel=string(vars[j]),
                    ylabel=string(vars[i]),
                    title="",
                    legend=false)
            end
            push!(plots, p)
        end

        plt = plot(plots...,
            layout=(n_cov, n_cov),
            size=(300 * n_cov, 300 * n_cov),
            plot_title="Covariate Scatter Matrix — $(name)")

        out_file = joinpath(figFolder, "covariate_scatter_matrix_$(name).png")
        savefig(plt, out_file)
        println("Saved $(out_file)")
        out_plots[string(name)] = plt
    end

    return out_plots
end

vars = [response, :credit_spr_lag1, :vix_lag1, :sentiment_lag1]
covariate_scatter_matrix(df_out, vars, periods)
covariate_scatter_matrix_multiple(df_out, vars, periods)

vars = [response, :cpi_infl_lag1, :sentiment, :log_oil]
covariate_scatter_matrix(df_out, vars, periods)
