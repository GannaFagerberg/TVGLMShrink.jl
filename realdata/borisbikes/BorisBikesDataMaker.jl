using CSV, DataFrames, Plots, Random, Distributions

dataFolder = joinpath(@__DIR__)

# read csv file and sink to dataframe
departure_df = CSV.read(joinpath(dataFolder, "Departure.csv"), DataFrame)
last(departure_df, 5)

# Plot all data
plot(departure_df.Date8h, departure_df."Waterloo Station 1, Waterloo",
    xlabel = "Time", ylabel = "Departures",
    title = "Waterloo Station 1",
    legend = false, lw = 1)

# Plot subset of data 
subTime = 3*30*12:3*30*12+100
plot(departure_df.Date8h[subTime], departure_df."Waterloo Station 1, Waterloo"[subTime],
    xlabel = "Time", ylabel = "Departures",
    title = "Waterloo Station 1",
    legend = false, lw = 1)

# Temperature data
tempdf = CSV.read(joinpath(dataFolder, "temp.csv"), DataFrame)
tempdf.t2m .-= 273.15 # air temp, Convert from Kelvin to Celsius
tempdf.d2m .-= 273.15 # dew point temp, Convert from Kelvin to Celsius

# Humidity
# Magnus formula for approx relative humidity (%) from air temp (T) and dew point (Td)
function relative_humidity(T, Td, b = 17.625, c = 243.04)
    return 100*exp(c*b*(Td-T)/((c+T)*(c+Td)))
end

humidity = relative_humidity.(tempdf.t2m, tempdf.d2m)

# rain
raindf = CSV.read(joinpath(dataFolder, "rain.csv"), DataFrame)
rain = raindf.tp*1000 # in mm

# sun
solardf = CSV.read(joinpath(dataFolder, "sun.csv"), DataFrame)
sun = solardf.ssrd / 3600 # Watt/m²

# snow
snowdf = CSV.read(joinpath(dataFolder, "snow.csv"), DataFrame)
snowcover = snowdf.snowc

# wind
winddf = CSV.read(joinpath(dataFolder, "wind.csv"), DataFrame)
wind_speed = sqrt.(winddf.u10.^2 .+ winddf.v10.^2)   # m/s
wind_dir = atan.(winddf.u10, winddf.v10) .* (180/π)  # meteorological convention may be adjusted

weather_df = DataFrame(datetime = tempdf.valid_time, temp  = tempdf.t2m, 
    humidity = humidity, rain = rain, sun = sun, snowcover = snowcover, 
    wind_speed = wind_speed)

CSV.write(joinpath(dataFolder, "weather_data.csv"), weather_df)

# Check against daily weather `$$ 
df_daily_weather = CSV.read(joinpath(dataFolder, "Weather(daily).csv"), DataFrame)

