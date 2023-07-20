using Printf
using CSV
using DataFrames
using Statistics
using Random
using Base.Threads
include("StateBuilder.jl")
include("Utils.jl")
include("Classifier.jl")

df = DataFrame(CSV.File("iris.csv"))

# adjustables
records = 1:100
features = 1:4
data_points = vcat(1:16, 51:66)
test_points = 1:100
epochs = 100

x = Matrix(df[records,features])
y = [df[i, 5] for i in records]

# standardize x
x = mapslices(col -> (col .- mean(col))./std(col), x, dims=1)
x[isnan.(x)] .= 0

# normalize x
x = mapslices(row -> row./norm(row), x, dims=2)
x[isnan.(x)] .= 0

# keyize y TODO: generalize
y = [y[i] == "setosa" ? -1 : 1 for i in 1:length(y)]

classifications = zeros(100)
for test_point in test_points
    sum = 0.0
    for data_point in data_points
        sum += y[data_point] * (1 - (1/(4*length(data_points)) * abs(x[test_point] .- x[data_point])^2))
    end
    classifications[test_point] = sum >= 0 ? 1 : -1
end

correct = 0
for test_point in test_points
    global correct
    if round(classifications[test_point]) == y[test_point]
        correct += 1
    end
end
print(correct / length(test_points), "\n")
