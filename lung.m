matlabpool open local
%% EXPERIMENT TWO - Using Any Data & Any Method %%

addpath('opls')

pheno = dataset('file','phenos.txt','delimiter',',','ReadVarNames',false);
pheno = double(pheno);

[xtrainAS, ytrainAS, gene_namesAS,isoform_namesAS,labelsAS, xtrainDE, ytrainDE, gene_namesDE, labelsDE, xtrainSNP, labelsSNP] = processData(pheno);
dsClust = load('data/avgCluster.mat');
dsClust = dsClust.avgCluster;
xtrainDE = dsClust;
xtrainDE = zscore(xtrainDE);

clusters = load('data/PCAclusters.mat');
clusters = clusters.avgCluster;
xtrainDE = clusters;

xtrainDE(19,:) = [];
ytrainDE(19) = [];
xtrainSNP(19,:) = [];

gene_namesSD = unique(gene_namesAS);
stanD = spliceMetric(gene_namesAS, xtrainAS);

% Designates method, data & selection f(n) to use in the GA
method = 'opls';   % Possible choices are: opls & svm
data = 'DE';      % Possible choices are: DE, AS, BOTH, SD, SNP & DESD
selection = @selectiontournament; % Possible choices are: @selectionremainder,@selectionuniform,{@selectionstochunif},@selectionroulette,@selectiontournament
cross = @crossovertwopoint; % Possible choices are: @crossoverheuristic,{@crossoverscattered},{@crossoverintermediate}*
                               % @crossoversinglepoint,@crossovertwopoint,@crossoverarithmetic
kernels = {'linear','quadratic','polynomial','rbf','mlp'};

% Initializes the "necessary" arrays (note the quotation marks around necessary)                             
results = {};
numCorrect = {};
numIncorrect = {};
numTimesInTop = {};
avgRank = {};
yPredTrain = {};
yPredTest = {};
yTrain = {};
topN = 100;
trainIndexes = {};
testIndexes = {};
allPops = {};
fvalues = {};
exits = {};
allOuts = {};
allScores = {};


% Need to include frequency of the gene showing up in top 20 (N)
nIterations = 250;
avgNumVariables = 0;
avgFracASVariables = 0;
desiredNGenes = 40;

DEgeneSubset = [];
ASgeneSubset = [];
DEGeneFreq = [];
ASGeneFreq = [];
totalGeneSubset = [];
bestChrSubset = [];
bestChrSubsetOPLS = [];
bestChrSubsetSVM = [];
gotWrong = [];
allPred = [];
classified = [];
bestFitnessScores = [];
bestFitnessSizes = [];
worstFitnessScores = [];
worstFitnessSizes = [];

fprintf('Starting Loop!\n'); 
% Iterates through the GA
parfor n = 1:nIterations
%     if (n>(nIterations/2))
%         method = 'svm';
%     end
    kernel = kernels(5);
    
    fprintf('Starting iteration %d\n',n);

    class1indexes = find(ytrainDE == 0);
    class2indexes = find(ytrainDE == 1);
    numSamples = length(ytrainDE);


    % Randomly picks one DF and one Relapse patient to leave out for testing
    i1 = randperm(length(class1indexes));
    i1 = [i1(1)];
    i2 = randperm(length(class2indexes));
    i2 = [i2(1)];
    leaveOutIndex = [class1indexes(i1);class2indexes(i2)];
    
    classified = [classified;leaveOutIndex];
   

    % Initializes the training and testing subsets
    xtrainSubsetDE = xtrainDE;
    xtrainSubsetDE(leaveOutIndex,:) = [];
    ytrainSubsetDE = ytrainDE;
    ytrainSubsetDE(leaveOutIndex,:) = [];
    xtestDE = xtrainDE(leaveOutIndex,:);
    ytestDE = ytrainDE(leaveOutIndex,:);

    xtrainSubsetAS = xtrainAS;
    xtrainSubsetAS(leaveOutIndex,:) = [];
    ytrainSubsetAS = ytrainAS;
    ytrainSubsetAS(leaveOutIndex,:) = [];
    xtestAS = xtrainAS(leaveOutIndex,:);
    ytestAS = ytrainAS(leaveOutIndex);
    
    xtrainSubsetSD = stanD;
    xtrainSubsetSD(leaveOutIndex,:) = [];
    xtestSD = stanD(leaveOutIndex,:);
    
    xtrainSubsetSNP = xtrainSNP;
    xtrainSubsetSNP(leaveOutIndex,:) = [];
    xtestSNP = xtrainSNP(leaveOutIndex, :);
    
    
    a = 1;  %%% made up value for opls num of factor

    % Initializes the fitness funcgtion and population information
    %fitness = @(member) (finalFitness(member,xtrainDE,ytrainDE, xtrainSubsetAS,ytrainSubsetAS, xtrainSNP, xtestSNP, xtestDE, ytestDE, xtestAS, ytestAS ,gene_namesDE,gene_namesAS,a,desiredNGenes,method, data, xtrainSubsetSD, xtestSD,kernel));
    fitness = @(member) (finalFitness(member,xtrainSubsetDE,ytrainSubsetDE, xtrainSubsetAS,ytrainSubsetAS, xtrainSubsetSNP, xtestSNP, xtestDE, ytestDE, xtestAS, ytestAS ,gene_namesDE,gene_namesAS,a,desiredNGenes,method, data, xtrainSubsetSD, xtestSD,kernel));
    popSize = 300;
    
    if (strcmp(data, 'DE'))  
        dataSize = size(xtrainSubsetDE,2);
        gene_names = gene_namesDE;
    elseif (strcmp(data, 'AS'))     
        dataSize = size(xtrainSubsetAS,2);
        gene_names = gene_namesAS;
    elseif (strcmp(data, 'BOTH'))
        dataSize = size(xtrainSubsetDE,2)+size(xtrainSubsetAS,2);
        gene_names = [gene_namesDE;gene_namesAS];
    elseif(strcmp(data, 'SD'))
        dataSize = size(xtrainSubsetSD,2);
        gene_names = unique(gene_namesAS);
    elseif(strcmp(data, 'DESD'))
        dataSize =  size(xtrainSubsetDE,2) + size(xtrainSubsetSD,2);
        gene_names = [gene_namesDE; unique(gene_namesAS)];
    elseif(strcmp(data, 'SNP'))
        dataSize = size(xtrainSubsetSNP,2);
        gene_names = labelsSNP;
    end
    
    initPop = zeros(popSize,dataSize);
    randomness = 10;

    % Creates the starting population
    for p = 1:popSize
        ixs = randperm(dataSize);
        initPop(p,ixs(1:round(desiredNGenes+rand*randomness))) = 1;
    end
    
    % Sets GA options and runs the GA
    options = gaoptimset('PopulationType','bitstring','initialpopulation',initPop, 'SelectionFcn', selection, 'CrossoverFcn', cross, 'populationsize',popSize);
    [x,fval,exitflag,output,population,scores]= gamultiobj(fitness,dataSize,[],[],[],[],[],[],options);
    % get outputs from MY model here
    
    % Stores the results from the GA run
    allPops{n} = population;
    fvalues{n} = fval;
    exits{n} = exitflag;
    allOuts{n} = output;
    allScores{n} = scores;
   
%     THIS RETURNS THE UNIQUE NAMES AND COUNTS OF ALL GENES IN A SINGLE
%     POPULATION, OR RUN, OR ALL DE GENES, ALL AS GENES, OR ALL GENES FROM
%     TOP CHROMOSOME FROM EACH RUN

        geneSubset = [];
        
        for i = 1:popSize
            if (strcmp(data, 'DE'))
            set = gene_names(find(population(i,:)));
            geneSubset = [geneSubset;set];
            
            elseif (strcmp(data, 'SD'))
            set = gene_names(find(population(i,:)));
            geneSubset = [geneSubset;set];
            
            elseif(strcmp(data, 'SNP'))
                set = gene_names(find(population(i,:)));
                geneSubset = [geneSubset;set];
            end

        end
        totalGeneSubset = [totalGeneSubset; geneSubset];

        [geneFreqNames, geneFreqNums] = count_unique(geneSubset);

        
        if (or(strcmp(data,'BOTH'), strcmp(data,'DESD')))
            DEGeneFreq = [DEGeneFreq; DEgeneSubset];
            ASGeneFreq = [ASGeneFreq;ASgeneSubset];
        end

% EXPORTS FILE CONTAINING GENE FREQUENCIES FROM EACH GA RUN
% df = dataset({},{},'VarNames',{'GeneNames','GeneNums'});
% df.GeneNames = geneFreqNames;
% df.GeneNums = geneFreqNums;
% exportName = strcat('19_gene_frequencies',num2str(n));
% export(df,'file',exportName);

% FINDS CHROMOSOME FROM SINGLE POPULATION WITH CLOSEST # OF GENES TO
% DESIRED
% minChrs = find(abs(scores(:,2)-desiredNGenes) == min(abs(scores(:,2) - desiredNGenes)));
% 
% if (length(minChrs) ~= 1)
%     bestChrs = find(scores(minChrs,1) == min(scores(minChrs,1)));
%     bestIndx = find(population(minChrs(bestChrs(1)),:));
%     bestChr = gene_names(bestIndx);
% else
% bestIndx = find(population(minChrs,:));
% bestChr = gene_names(bestIndx);
% end
% 
% bestChrSubset = [bestChrSubset; bestChr];
% if (strcmp(method,'opls'))
%     bestChrSubsetOPLS = [bestChrSubsetOPLS; bestChr];
% else
%     bestChrSubsetSVM = [bestChrSubsetSVM; bestChr];
% end

        geneSubset = [];
        counts = [];
        for i = 1:popSize

            set = gene_names(find(population(i,:)));
            geneSubset = [geneSubset;set];
            counts(i) = length(find(population(i,:)));

        end
        totalGeneSubset = [totalGeneSubset; geneSubset];

        [geneFreqNames, geneFreqNums] = count_unique(geneSubset);
        

        bestChrInx = find(scores(:,1) == min(scores(:,1)));
        
        bestChrMin = min(counts(bestChrInx));
        
        best = find(counts(bestChrInx) == bestChrMin);
        best = bestChrInx(best(1));
        bestIndx = find(population(best,:));
        bestGenes = gene_names(find(population(best,:)));
        
        bestChrSubset = [bestChrSubset;bestGenes];
        
        if (strcmp(method,'opls'))
            bestChrSubsetOPLS = [bestChrSubsetOPLS; bestGenes];
        else
            bestChrSubsetSVM = [bestChrSubsetSVM; bestGenes];
        end
        
bestFitnessScores = [bestFitnessScores;scores(best,1)];
bestFitnessSizes = [bestFitnessSizes;scores(best,2)];

    worstChrInx = find(scores(:,1) == max(scores(:,1)));

    worstChrMin = max(counts(worstChrInx));

    worst = find(counts(worstChrInx) == worstChrMin);
    worst = worstChrInx(worst(1));
        
worstFitnessScores = [worstFitnessScores;scores(worst,1)];
worstFitnessSizes = [worstFitnessSizes;scores(worst,2)];


% CALCULATES ACCURACY FOR OPLS MODEL

if (strcmp(data, 'DE'))
    xtrain = xtrainSubsetDE;
    xtest = xtestDE;
elseif (strcmp(data, 'AS'))
    xtrain = xtrainSubsetAS;
    xtest = xtestAS;
elseif (strcmp(data, 'BOTH'))
    xtrain = [xtrainSubsetDE,xtrainSubsetAS];
    xtest = [xtestDE,xtestAS];           
elseif(strcmp(data, 'SD'))
    xtrain = xtrainSubsetSD;
    xtest = xtestSD;
elseif(strcmp(data, 'DESD'))
   xtrain = [xtrainSubsetDE,xtrainSubsetSD];
   xtest = [xtestDE,xtestSD];
elseif(strcmp(data, 'SNP'))
    xtrain = xtrainSubsetSNP;
    xtest = xtestSNP;
end
    
    predicted = zeros(1, length(ytestDE));
    TrainSubset = xtrain(:,bestIndx);
    TestSubset = xtest(:,bestIndx);
if (strcmp(method,'opls'))

    [model, stats] = opls(TrainSubset, ytrainSubsetDE, a);
    [t,t_ortho,Y_pred] = apply_opls_model(TrainSubset,ytrainSubsetDE,model,TestSubset);
    Y_pred = round(Y_pred);
    for i = 1:length(Y_pred)
        if (Y_pred(i) == ytestDE(i))
            predicted(i) = 1;
        else
            gotWrong = [gotWrong;leaveOutIndex(i)];
        end
    end
    predicted = sum(predicted) / length(predicted);
    predicted = sum(predicted) / length(predicted);
    allPred = [allPred;predicted];
    fprintf('OPLS Prediction Accuracy for top chromosome from run # %d using %s = %d\n', n,data,predicted);
% CALCULATED ACCURACY FOR SVM MODEL
else
    svmStruct = svmtrain(TrainSubset, ytrainSubsetDE, 'kernel_function', 'rbf');
    [class, SVMscores] = ga_svm_classify(svmStruct, TestSubset);
    for i = 1:length(ytestDE)
        if (class(i) == ytestDE(i))
            predicted(i) = 1;
        else
            gotWrong = [gotWrong;leaveOutIndex(i)];

        end
    end
    predicted = sum(predicted) / length(predicted);
    allPred = [allPred;predicted];
    
    disp(strcat('SVM Prediction Accuracy for top chromosome from run #', num2str(n), ' using ->', data, ' = ', num2str(predicted)));
    
end
end

allPred = sum(allPred) / nIterations;
[wrongIndx, wrongCounts] = count_unique(gotWrong);
[classifiedNums, classifiedCounts] = count_unique(classified);
[topGeneFreqNames, topGeneFreqNums] = count_unique(bestChrSubset);
clustIndx = zeros(length(topGeneFreqNames),1);

for i = 1:length(topGeneFreqNames)
num = find(ismember(gene_namesDE,topGeneFreqNames(i)));
clustIndx(i) = num;
end



% EXPORTS ALL GENE FREQUENCIES OUT INTO TXT FILES --> USED PRIMARILY FOR
% CLUSTER RUNS
df = dataset({},{},'VarNames',{'ClassifiedNums','ClassfiedCounts'});
df.ClassifiedNums = classifiedNums;
df.ClassifiedCounts = classifiedCounts;
export(df,'file','avg_classified_freq.txt');

df = dataset({},{},'VarNames',{'wrongSamples','wrongCounts'});
df.wrongSamples = wrongIndx;
df.wrongCounts = wrongCounts;
export(df,'file','avg_wrong_frequencies.txt');

df = dataset({}, 'VarNames',{'PredictPercent'});
df.PredictPercent = allPred;
export(df,'file','avg_predict_percent.txt');

df = dataset({},{},'VarNames',{'TopGeneNames','TopGeneNums'});
df.TopGeneNames = clustIndx;
df.TopGeneNums = topGeneFreqNums;
export(df,'file','avg_top_ALL_frequencies.txt');

bestFitnessAverage = mean(bestFitnessScores);
tdf = dataset({},{},{},'VarNames',{'bestFitnessValues','bestFitnessSizes','bestFitnessAvg'});
tdf.bestFitnessValues = bestFitnessScores;
tdf.bestFitnessSizes = bestFitnessSizes;
tdf.bestFitnessAvg = bestFitnessAverage;
export(tdf,'file','avg_best_fitness_values.txt');

worstFitnessAverage = mean(worstFitnessScores);
tdf = dataset({},{},{},'VarNames',{'worstFitnessValues','worstFitnessSizes','worstFitnessAvg'});
tdf.worstFitnessValues = worstFitnessScores;
tdf.worstFitnessSizes = worstFitnessSizes;
tdf.worstFitnessAvg = worstFitnessAverage;
export(tdf,'file','avg_worst_fitness_values.txt');





matlabpool close
