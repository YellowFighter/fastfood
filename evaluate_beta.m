function metrics = evaluate_beta(B,Bfit)

metrics = {};

metrics.MAE = mean(abs(B-Bfit));
zeroInxs = find(B == 0);
metrics.MAEzeros = mean(abs(B(zeroInxs)-Bfit(zeroInxs)));
metrics.numCorrectZeros = length(find(B == 0 & Bfit == 0));
metrics.perCorrectZeros = metrics.numCorrectZeros/length(zeroInxs);

