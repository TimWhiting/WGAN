using stats

# These are the losses for the sNORB_mlp_dcgan model,
# for epochs 1-79. Each entry is of the format:
# ["criticLoss", "generatorloss"].
# One entry for each epoch.
losses = [
    [0.0527, 0.0214],
    [0.0535, 0.0196],
    [0.0528, 0.0189],
    [0.0523, 0.0192],
    [0.0516, 0.0184],
    [0.0516, 0.0183],
    [0.0531, 0.0197],
    [0.0521, 0.0188],
    [0.0519, 0.0190],
    [0.0523, 0.0195],
    [0.0515, 0.0184],
    [0.0523, 0.0187],
    [0.0531, 0.0190],
    [0.0536, 0.0190],
    [0.0544, 0.0198],
    [0.0545, 0.0196],
    [0.0545, 0.0196],
    [0.0550, 0.0195],
    [0.0533, 0.0191],
    [0.0547, 0.0195],
    [0.0548, 0.0195],
    [0.0551, 0.0195],
    [0.0560, 0.0206],
    [0.0546, 0.0195],
    [0.0541, 0.0190],
    [0.0548, 0.0197],
    [0.0550, 0.0195],
    [0.0541, 0.0191],
    [0.0540, 0.0193],
    [0.0529, 0.0190],
    [0.0515, 0.0178],
    [0.0418, 0.0164],
    [0.0212, 0.0166],
    [0.0239, 0.0224],
    [0.0264, 0.0247],
    [0.0223, 0.0245],
    [0.0269, 0.0320],
    [0.0235, 0.0264],
    [0.0204, 0.0258],
    [0.0185, 0.0175],
    [0.0525, 0.0541],
    [0.0606, 0.0588],
    [0.0609, 0.0638],
    [0.0455, 0.0448],
    [0.0203, 0.0161],
    [0.0250, 0.0223],
    [0.0538, 0.0517],
    [0.0374, 0.0395],
    [0.0478, 0.0403],
    [0.0232, 0.0217],
    [0.0418, 0.0350],
    [0.0403, 0.0391],
    [0.0559, 0.0558],
    [0.0248, 0.0230],
    [0.0541, 0.0523],
    [0.0313, 0.0296],
    [0.0258, 0.0213],
    [0.0354, 0.0364],
    [0.0350, 0.0378],
    [0.0196, 0.0177],
    [0.0238, 0.0196],
    [0.0647, 0.0617],
    [0.0470, 0.0426],
    [0.0385, 0.0359],
    [0.0397, 0.0384],
    [0.0254, 0.0227],
    [0.0382, 0.0395],
    [0.0426, 0.0373],
    [0.0256, 0.0230],
    [0.0542, 0.0511],
    [0.0435, 0.0389],
    [0.0583, 0.0529],
    [0.0619, 0.0621],
    [0.0561, 0.0560],
    [0.0432, 0.0429],
    [0.0349, 0.0307],
    [0.0399, 0.0429],
    [0.0438, 0.0451],
    [0.0209, 0.0151]
]

# Build object
ganStats = GANStats()
for lossEntry in losses
    push!(ganStats.cLoss, lossEntry[1])
    push!(ganStats.gLoss, lossEntry[2])
end

# Plot results

plotGANStats(ganStats, "sNORB_mlp_dcgan_v2_loss_1-79")