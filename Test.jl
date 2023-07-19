using Printf
using CSV
using DataFrames
using Statistics
using Random
include("StateBuilder.jl")
include("Utils.jl")
include("Classifier.jl")

df = DataFrame(CSV.File("iris.csv"))

# raw data
#records = vcat(1:32, 51:82)
records = 1:100
features = 1:4
x = Matrix(df[records,features])
y = [df[i, 5] for i in records]

# standardize x
x = mapslices(col -> (col .- mean(col))./std(col), x, dims=1)
x[isnan.(x)] .= 0

# normalize x
x = mapslices(row -> row./norm(row), x, dims=2)
x[isnan.(x)] .= 0
cheese = copy(x)

# keyize y TODO: generalize
y = [y[i] == "setosa" ? 1 : 0 for i in 1:length(y)]

# test Classifier.jl
classifications = []
for test_point_index in 1:100
    push!(classifications, classify(x[vcat(1:32, 51:82),:],y[vcat(1:32, 51:82)],x[33,:],100))
    print("point ", test_point_index, " ", classifications[test_point_index], " ", y[33], "\n")
end
