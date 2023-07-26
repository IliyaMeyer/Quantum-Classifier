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
features = 1:2
data_points = vcat(21:36, 81:961)
test_points = [50]
epochs = 100

x = Matrix(df[records,features])
y = [df[i, 5] for i in records]

# standardize x
#x = mapslices(col -> (col .- mean(col))./std(col), x, dims=1)
x[isnan.(x)] .= 0

# normalize x
x = mapslices(row -> row./norm(row), x, dims=2)
x[isnan.(x)] .= 0

# keyize y
y = [y[i] == "setosa" ? 0 : 1 for i in 1:length(y)]

# test Classifier.jl
classifications = zeros(100)
@time begin
    @threads for test_point_index in test_points
        classifications[test_point_index] = classify(x[data_points,:],y[data_points],x[test_point_index,:],epochs)
        print("point ", test_point_index, " ", classifications[test_point_index], " ", y[test_point_index], "\n")
    end
end

correct = 0
@threads for test_point in test_points
    global correct
    if round(classifications[test_point]) == y[test_point]
        correct += 1
    end
end
print(correct / length(test_points), "\n")
