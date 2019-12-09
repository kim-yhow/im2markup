function Rowtransfer(use_cuda)
    local model = nn.Sequential()
    model:add(nn.SplitTable(1, 3)) -- #H list of (batch_size, W, 512)
    return model
end

--(batch_size, H, W, 512)

function Coltransfer(use_cuda)
    local model = nn.Sequential()
    model:add(nn.SplitTable(2, 3)) -- #W list of (batch_size, W, 512)
    return model
end

