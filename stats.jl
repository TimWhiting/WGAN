module stats

using Plots

mutable struct LearningStats
    trainAcc::Array{Float64,1}
    trainLoss::Array{Float64,1}
    valAcc::Array{Float64,1}
    valLoss::Array{Float64,1}
    testAcc::Array{Float64,1}
    testLoss::Array{Float64,1}
    bestValAcc::Float64
    bestValLoss::Float64
end
LearningStats() = LearningStats([], [], [], [], [], [], -Inf64, Inf64)

function plotLearningStats(stats::LearningStats, name::String, isClassification::Bool)
    if isClassification
        plt = plot(1:length(stats.valLoss), hcat(stats.trainLoss, stats.valLoss, stats.valAcc), label = ["Train Loss", "Val. Loss", "Val. Accuracy"], xlabel = "Epochs", ylabel =  "Loss")
    else
        plt = plot(1:length(stats.train), hcat(stats.trainLoss, stats.valLoss), label = ["Train Loss", "Val. Loss"], xlabel = "Epochs", ylabel = "Loss")
    end
    savefig(plt, "$(name).png");
end

export LearningStats, plotLearningStats

end # module stats