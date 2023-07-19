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
records = vcat(1:32, 52:83)
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

num_records = size(x, 1)
num_features = size(x, 2)
print(@sprintf("Data points:\t%d\nFeatures:\t%d\n", num_records, num_features))

# test Classifier.jl
test_point_index = rand(1:length(records))
print(classify(x,y,x[33,:],100), " ", y[33])
