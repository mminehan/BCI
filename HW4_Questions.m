%%
% <latex> \title{BE 521: Homework 4 \\{\normalsize HFOs}
% \\{\normalsize Spring 2016}} \author{58 points} \date{Due: Tuesday,
% 2/23/2016 11:59pm} \maketitle \textbf{Objective:} HFO detection and
% cross-validation </latex>

%% 
% <latex> \begin{center} \author{Michael Minehan \\
%   \normalsize Collaborators: Chang Su, Sung Min Ha, Danny Lin \\}
% \end{center} 
% </latex>


%% 
% <latex>
% \subsection*{HFO Dataset} High frequency oscillations (HFOs) are
% quasi-periodic intracranial EEG transients with durations on the
% order of tens of milliseconds and peak frequencies in the range of
% 80 to 500 Hz. There has been considerable interest among the
% epilepsy research community in the potential of these signals as
% biomarkers for epileptogenic networks.\\\\
% In this homework exercise, you will explore a dataset of candidate
% HFOs detected using the algorithm of Staba et al. (see article on
% Canvas). The raw recordings from which this dataset arises come from
% a human subject with mesial temporal lobe epilepsy and were
% contributed by the laboratory of Dr. Greg Worrell at the Mayo Clinic
% in Rochester, MN.\\\\
% The dataset \verb|I521_A0004_D001| contains raw HFO clips that are
% normalized to zero mean and unit standard deviation but are
% otherwise unprocessed. The raw dataset contain two channels of data:
% \verb|Test_raw_norm| and \verb|Train_raw_norm|, storing raw testing
% and training sets of HFO clips respectively. The raw dataset also
% contains two annotation layers: \verb|Testing windows| and
% \verb|Training windows|, storing HFO clip start and stop times (in
% microseconds) for each of the two channels above.
% Annotations contain the classification by an ``expert'' reviewer
% (i.e., a doctor) of each candidate HFO as either an HFO (2) or an
% artifact (1). On ieeg.org and upon downloading the annotations, 
% You can view this in the "description" field. \\\\
% After loading the dataset in to a \verb|session| variable as in
% prior assignments you will want to familiarize yourself with the
% \verb|IEEGAnnotationLayer| class. Use the provided "getAnnotations.m"
% function to get all the annotations from a given dataset. The first
% output will be an array of annotation objects, which you will see
% also has multiple fields including a description field as well as start and stop times. Use
% You can use the information outputted by getAnnotations to pull
% each HFO clip.
% </latex>

addpath(genpath('../ieeg-matlab-1.13.2'));
session = IEEGSession('I521_A0004_D001', 'minehan', 'min_ieeglogin.bin')

session.openDataSet('I521_A0004_D001');

sampleRate_A4D1 = session.data(1).sampleRate;

nr = ceil((session.data(1).rawChannels(1).get_tsdetails.getEndTime)...
    /1e6*sampleRate_A4D1);
nr2 = ceil((session.data(1).rawChannels(2).get_tsdetails.getEndTime)...
    /1e6*sampleRate_A4D1);
allData_A4D1 = session.data(1).getvalues(1:nr,1)';
allData_A4D1_train = session.data(1).getvalues(1:nr2,2)';
time_A4D1 = (1:nr)/sampleRate_A4D1;
time_A4D1_train = (1:nr2)/sampleRate_A4D1;
save('allData_A4D1','allData_A4D1')
save('allData_A4D1_train','allData_A4D1_train')
save('time_A4D1','time_A4D1')
save('time_A4D1_train','time_A4D1_train')
save('sampleRate_A4D1','sampleRate_A4D1')


load('allData_A4D1')
load('allData_A4D1_train')
load('time_A4D1')
load('time_A4D1_train')
load('sampleRate_A4D1')

addpath(genpath('../ieeg-matlab-1.13.2'));
session = IEEGSession('I521_A0004_D001', 'minehan', 'min_ieeglogin.bin')
[allEvents, timesUSec, channels] = getAnnotations(session.data(1),'Training windows')

plot(time_A4D1_train,allData_A4D1_train)
%% 
% <latex>
% \section{Simulating the Staba Detector (12 pts)} Candidate HFO clips
% were detected with the Staba et al. algorithm and subsequently
% validated by an expert as a true HFO or not. In this first section,
% we will use the original iEEG clips containing HFOs and re-simulate
% a portion of the Staba detection.
% \begin{enumerate}
%     \item How many samples exist for each class (HFO vs artifact) in
%     the training set? (Show code to support your answer) (1 pt)
% </latex>

%%

% Extract classification of each event
for jj = 1:numel(allEvents);
    type(1,jj) = str2num(allEvents(jj).description);
end

% Determine number of artifacts and number of HFOs
num_arts = numel(type((type == 1)))
num_hfos = numel(type((type == 2)))
%% 
% <latex>
%     \item Using the training set, find the first occurrence of the
%     first valid HFO and the first artifact.
%         Using \verb|subplot| with 2 plots, plot the valid HFO's
%         (left) and artifact's (right) waveforms. Since the units are
%         normalized, there's no need for a y-axis, so remove it with
%         the command \verb|set(gca,'YTick',[])|. (2 pts)
% </latex>

%%

% Extract start and stop index for all events
start_t = zeros(1,numel(allEvents));
stop_t = zeros(1,numel(allEvents));
for ii = 1:numel(allEvents);
    start_t(1,ii) = allEvents(ii).start;
    stop_t(1,ii) = allEvents(ii).stop;
end

artIn = find(type == 1);
hfoIn = find(type == 2);

fs = sampleRate_A4D1;

start_ind = ceil((start_t*fs)/1e6);
stop_ind = ceil((stop_t*fs)/1e6);

firstArt = allData_A4D1_train(start_ind(artIn(1)):stop_ind(artIn(1)));
firstHFO = allData_A4D1_train(start_ind(hfoIn(1)):stop_ind(hfoIn(1)));
firstArt_t = time_A4D1_train(start_ind(artIn(1)):stop_ind(artIn(1)));
firstHFO_t = time_A4D1_train(start_ind(hfoIn(1)):stop_ind(hfoIn(1)));


figure(1)
subplot(1,2,1)
plot(firstHFO_t, firstHFO)
set(gca,'YTick',[])
xlabel('Time (s)')
ylabel('HFO')
subplot(1,2,2)
plot(firstArt_t, firstArt)
set(gca,'YTick',[])
xlabel('Time (s)')
ylabel('Artifact')
%% 
% <latex>
%     \item Using the \texttt{fdatool} in MATLAB, build an FIR
%     bandpass filter of the equiripple type of order 100.
%         Use the Staba et al. (2002) article to guide your choice of
%         passband and stopband frequency. Once set, go to
%         \texttt{File} \verb|->| \texttt{Export}, and export
%         ``Coefficients'' as a MAT-File. Attach a screenshot of your
%         filter's magnitude response. (Note: We will be flexible with
%         the choice of frequency parameters within reason.) (3 pts)
% </latex>

%%
figure(2)
imread('MagnitudeResponse.png');
imshow('MagnitudeResponse.png')

%% 
% <latex>
%     \item Using the forward-backward filter function
%     (\texttt{filtfilt}) and the numerator coefficients saved above,
%         filter the valid HFO and artifact clips obtained earlier.
%         You will need to make a decision about the input argument
%         \texttt{a} in the \texttt{filtfilt} function. Plot these two
%         filtered clips overlayed on their original signal in a two
%         plot \texttt{subplot} as before. Remember to remove the
%         y-axis. (3 pts)
% </latex>

%%

load('Staba_Bandpass')
filt_HFO = filtfilt(Num,1,firstHFO);
filt_Art = filtfilt(Num,1,firstArt);

% Testing a different set of pass and stop frequencies
% % load('Test_filter')
% % filt_HFO = filtfilt(Num1,1,firstHFO);
% % filt_Art = filtfilt(Num1,1,firstArt);

figure(3)
subplot(1,2,1)
hold on
plot(firstHFO_t, filt_HFO)
plot(firstHFO_t, firstHFO,'r')
set(gca,'YTick',[])
xlabel('Time (s)')
ylabel('HFO Amplitude')
legend('Filtered HFO','Raw HFO')
hold off
subplot(1,2,2)
hold on
plot(firstArt_t, filt_Art)
plot(firstArt_t, firstArt,'r')
set(gca,'YTick',[])
xlabel('Time (s)')
ylabel('Artifact Amplitude')
legend('Filtered Artifact','Raw Artifact')
hold off
%% 
% <latex>
%     \item Speculate how processing the data using Staba's method may
%     have erroneously led to a false HFO detection (3 pts)
% </latex>

%%

% Artifacts often contain sharp changes in the time domain, which, after
% filtering, translate to high frequency rippling in the time domain.
% These high frequency ripples are difficult to distinguish from
% legitimate HFOs, which could lead to false HFO detection fairly easily.

%% 
% <latex>
% \end{enumerate}
% \section{Defining Features for HFOs (9 pts)} In this section we will
% be defining a feature space for the iEEG containing HFOs and
% artifacts. These features will describe certain attributes about the
% waveforms upon which a variety of classification tools will be
% applied to better segregate HFOs and artifacts
% \begin{enumerate}
%     \item Create two new matrices, \verb|trainFeats| and
%     \verb|testFeats|, such that the number of rows correspond to
%     observations (i.e. number of training and testing clips)
%         and the number of columns is two. Extract the line-length
%         and area features (seen previously in lecture and Homework
%         3) from the normalized raw signals. Store the line-length
%         value in the first column and area value for each sample in
%         the second column of your features matrices. Make a scatter
%         plot of the training data in the 2-dimensional feature
%         space, coloring the valid detections blue and the artifacts
%         red. (Note: Since we only want one value for each feature of
%         each clip, you will effectively treat the entire clip as the
%         one and only ``window''.) (4 pts)
% </latex>

%%

%Create anonymous function
LLFn = @(x) sum(abs(diff(x)));
area = @(x) sum(abs(x));

addpath(genpath('../ieeg-matlab-1.13.2'));
session = IEEGSession('I521_A0004_D001', 'minehan', 'min_ieeglogin.bin')
[testEvents, timesUSec_test, channels_test] = getAnnotations(session.data(1),'Testing windows')

start_t_test = zeros(1,numel(testEvents));
stop_t_test = zeros(1,numel(testEvents));
for ii = 1:numel(testEvents);
    start_t_test(1,ii) = testEvents(ii).start;
    stop_t_test(1,ii) = testEvents(ii).stop;
end

start_ind_test = ceil((start_t_test*fs)/1e6);
stop_ind_test = ceil((stop_t_test*fs)/1e6);

%Call anonymous function

LL_train = zeros(1,length(allEvents));
A_train = zeros(1,length(allEvents));
LL_test = zeros(1,length(testEvents));
A_test = zeros(1,length(testEvents));

for kk = 1:length(allEvents)
    
    LL_train(kk) = LLFn(allData_A4D1_train(start_ind(kk):stop_ind(kk)));
    A_train(kk) = area(allData_A4D1_train(start_ind(kk):stop_ind(kk)));
end

for ll = 1:length(testEvents)
    
    LL_test(ll) = LLFn(allData_A4D1(start_ind_test(ll):stop_ind_test(ll)));
    A_test(ll) = area(allData_A4D1(start_ind_test(ll):stop_ind_test(ll)));
end
trainFeats = [LL_train', A_train'];
testFeats = [LL_test', A_test'];

trainArt = NaN*ones(size(trainFeats));
trainHFO = NaN*ones(size(trainFeats));
for ff = 1:numel(allEvents)
    if type(ff) == 1
        trainArt(ff,:) = trainFeats(ff,:);
    else
        trainHFO(ff,:) = trainFeats(ff,:);
    end
end

figure(4)
hold on
scatter(trainArt(:,1),trainArt(:,2),'r')
scatter(trainHFO(:,1),trainHFO(:,2),'b')
xlabel('Line Length')
ylabel('Area')
hold off
%% 
% <latex>
%     \item Feature normalization is often important. One simple
%     normalization method is to subtract each feature by its mean and
%     then divide by its standard deviation
%         (creating features with zero mean and unit variance). Using
%         the means and standard deviations calculated in your
%         \emph{training} set features, normalize both the training
%         and testing sets.
%     \begin{enumerate} \item What is the statistical term for the
%     normalized value, which you have just computed? (1 pt)
% </latex>

%%
% This is a z-score

featsMeans_train = mean(trainFeats,1)
tFeatMeansXtended = [featsMeans_train(1,1)*ones(length(trainFeats),1), featsMeans_train(1,2)*ones(length(trainFeats),1)];
x_minus_mean = (trainFeats - tFeatMeansXtended);
std_train = std(trainFeats);
normFeats_train = [x_minus_mean(:,1)/std_train(1,1), x_minus_mean(:,2)/std_train(1,2)];

%% 
% <latex>
% 	\item Explain why such a feature normalization might be critical
% 	to the performance of a $k$-NN classifier. (2 pts) 
% </latex>

%%
% Feature normalization is essential in order to compare values between
% different features. Without this, one feature may unfairly dominate the
% classifying function.

%%
% <latex>
% \item Explain why (philosophically) you use the training feature means and
% 	standard deviations to normalize the testing set. (2 pts)
% </latex>

%%
% Using the means and standard deviations of the training features is
% important to show that the training data is appropriate for training a
% classification algorithm. If the means and standard deviations of the
% training set varied wildly from what could reasonably be expected in a
% sample, it would be a poor training set.

%% 
% <latex>
%     \end{enumerate}
% \end{enumerate}
% \section{Comparing Classifiers (20 pts)} In this section, you will
% explore how well a few standard classifiers perform on this dataset.
% Note, the logistic regression and $k$-NN classifiers are functions
% built into some of Matlabs statistics packages. If you don't have
% these (i.e., Matlab doesn't recognize the functions), we've provided
% them, along with the LIBSVM mex files, in the \verb|lib.zip| file.
% To use these, unzip the folder and add it to your Matlab path with
% the command \texttt{addpath lib}. If any of these functions don't
% work, please let us know.
% \begin{enumerate}
%  \item Using Matlab's logistic regression classifier function,
%  (\texttt{mnrfit}), and its default parameters, train a model on the
%  training set. Using Matlab's \texttt{mnrval} function, calculate
%  the training error (as a percentage) on the data. For extracting
%  labels from the matrix of class probabilities, you may find the
%  command \texttt{[$\sim$,Ypred] = max(X,[],2)} useful\footnote{Note:
%  some earlier versions of Matlab don't like the \texttt{$\sim$},
%  which discards an argument, so just use something like
%  \texttt{[trash,Ypred] = max(X,[],2)} instead.}, which gets the
%  column-index of the maximum value in each row (i.e., the class with
%  the highest predicted probability). (3 pts)
% </latex>

%%
% 
addpath('libsvmmatlab')

reg_coeff = mnrfit(normFeats_train, type');
pihat = mnrval(reg_coeff,normFeats_train);
[~,Ypred] = max(pihat,[],2);

% Error as a percent
trainingError = 100*sum(abs(Ypred - type'))/numel(type)


%% 
% <latex>
%  \item Using the model trained on the training data, predict the
%  labels of the test samples and calculate the testing error. Is the
%  testing error larger or smaller than the training error? Give one
%  sentence explaining why this might be so. (2 pts)
% </latex>

%%
% 
addpath('libsvmmatlab')

type_test = zeros(1,length(testEvents));
for jj = 1:numel(testEvents);
    type_test(1,jj) = str2num(testEvents(jj).description);
end

testFeatMeansXtended = [featsMeans_train(1,1)*ones(length(testFeats),1), featsMeans_train(1,2)*ones(length(testFeats),1)];
x_minus_mean_test = (testFeats - testFeatMeansXtended);
normFeats_test = [x_minus_mean_test(:,1)/std_train(1,1), x_minus_mean_test(:,2)/std_train(1,2)];

% reg_coeff_test = mnrfit(normFeats_test, type_test');
pihat_test = mnrval(reg_coeff_test,normFeats_test);
[~,Ypred_test] = max(pihat_test,[],2);

% Error as a percent
testingError = 100*sum(abs(Ypred_test - type_test'))/numel(type_test)

% This is larger than the training error. The regression
% was performed on the training data, which optimized the regression's fit
% for the training data, not the testing data. We therefore expect the
% regression model applied to the testing data to have a larger calculated
% error than the same model applied to its own training data.

%% 
% <latex>
%  \item
%   \begin{enumerate}
%    \item Use Matlab's $k$-nearest neighbors function,
%    \texttt{knnclassify}, and its default parameters ($k$ = 1, among
%    other things), calculate the training and testing errors. (3 pts)
%    \item Why is the training error zero? (2 pts)
% </latex>

%%
% 

knn_train = knnclassify(normFeats_train, normFeats_train, type);
knn_test = knnclassify(normFeats_test, normFeats_train, type);

kTrainingError = 100*sum(abs(knn_train - type'))/numel(type)
kTestingError = 100*sum(abs(knn_test - type_test'))/numel(type_test)


% The training error is zero because each event is assigned its identity as
% an artifact or HFO based on the assignment that each event already had
% inputted into the function. In other words, no reassignments are made, so
% the classifications produced match the classifications inputted exactly.
%% 
% <latex>
%   \end{enumerate}
%  \item In this question you will use the LIBSVM implementation of a
%  support vector machine (SVM). LIBSVM is written in C, but we'll use
%  the Matlab executable versions (with *.mex* file extensions). Type
%  \texttt{svmtrain} and \texttt{svmpredict} to see how the functions
%  are used\footnote{Matlab has its own analogous functions,
%  \texttt{svmtrain} and \texttt{svmclassify}, so make sure that the
%  LIBSVM files have been added to your path (and thus will superceed
%  the default Matlab functions).}. Report the training and testing
%  errors on an SVM model with default parameters. (3 pts)
% </latex>

%%
%

svm_model = svmtrain(type', normFeats_train, 0);
[predicted_label_train, accuracy_train, decision_values_train] = svmpredict(type', normFeats_train, svm_model, 0);
[predicted_label_test, accuracy_test, decision_values_test] = svmpredict(type_test', normFeats_test, svm_model, 0);

svm_train_error = accuracy_train(2)*100
svm_test_error = accuracy_test(2)*100
%% 
% <latex>
%  \item It is sometimes useful to visualize the decision boundary of
%  a classifier. To do this, we'll plot the classifier's prediction
%  value at every point in the ``decision'' space. Use the
%  \texttt{meshgrid} function to generate points in the line-length
%  and area 2D feature space and a scatter plot (with the \verb|'.'|
%  point marker) to visualize the classifier decisions at each point
%  (use yellow and cyan for your colors). In the same plot, show the
%  training samples (plotted with the '*' marker to make them more
%  visible) as well. As before use blue for the valid detections and
%  red for the artifacts. Use ranges of the features that encompass
%  all the training points and a density that yields that is
%  sufficiently high to make the decision boundaries clear. Make such
%  a plot for the logistic regression, $k$-NN, and SVM classifiers. (4
%  pts)
% </latex>

%%
% 
LLrange = min(normFeats_train(:,1)):0.04:max(normFeats_train(:,1));
Arange = min(normFeats_train(:,2)):0.05:max(normFeats_train(:,2));
[LLspace, Aspace] = meshgrid(LLrange,Arange);
space = [LLspace(:) Aspace(:)];

% Logistic
coeff = mnrfit(normFeats_train, type');
pihat_logspace = mnrval(coeff,space);
[~,Ypred_logspace] = max(pihat_logspace,[],2);

spaceArt = NaN*ones(length(space),2);
spaceHFO = NaN*ones(length(space),2);
for ff = 1:length(space)
    if Ypred_logspace(ff) == 1
        spaceArt(ff,:) = space(ff,:);
    elseif Ypred_logspace(ff) == 2
        spaceHFO(ff,:) = space(ff,:);
    else
        break
    end
end

trainArtN = NaN*ones(length(type),2);
trainHFON = NaN*ones(length(type),2);
for ff = 1:numel(allEvents)
    if type(ff) == 1
        trainArtN(ff,:) = normFeats_train(ff,:);
    else
        trainHFON(ff,:) = normFeats_train(ff,:);
    end
end

figure(5)
hold on
scatter(spaceArt(:,1),spaceArt(:,2),'.','y')
scatter(spaceHFO(:,1),spaceHFO(:,2),'.','c')
scatter(trainArtN(:,1),trainArtN(:,2),'r')
scatter(trainHFON(:,1),trainHFON(:,2),'b')
title('Logistical Regression')
hold off

% knn

knn = knnclassify(space, normFeats_train, type);
spaceArt_knn = NaN*ones(length(space),2);
spaceHFO_knn = NaN*ones(length(space),2);
for ff = 1:length(space)
    if knn(ff) == 1
        spaceArt_knn(ff,:) = space(ff,:);
    elseif knn(ff) == 2
        spaceHFO_knn(ff,:) = space(ff,:);
    else
        break
    end
end

figure(6)
hold on
scatter(spaceArt_knn(:,1),spaceArt_knn(:,2),'.','y')
scatter(spaceHFO_knn(:,1),spaceHFO_knn(:,2),'.','c')
scatter(trainArtN(:,1),trainArtN(:,2),'r')
scatter(trainHFON(:,1),trainHFON(:,2),'b')
title('knn')
hold off

% SVM

placeholder = zeros(length(space),1);

model = svmtrain(type', normFeats_train, 0);
[predicted_label_train, accuracy_train, decision_values_train] = svmpredict(type', normFeats_train, model, 0);
[predicted_label_space, accuracy_space, decision_values_space] = svmpredict(placeholder, space, model, 0);

spaceArt_svm = NaN*ones(length(space),2);
spaceHFO_svm = NaN*ones(length(space),2);
for ff = 1:length(space)
    if predicted_label_space(ff) == 1
        spaceArt_svm(ff,:) = space(ff,:);
    elseif predicted_label_space(ff) == 2
        spaceHFO_svm(ff,:) = space(ff,:);
    else
        break
    end
end

figure(7)
hold on
scatter(spaceArt_svm(:,1),spaceArt_svm(:,2),'.','y')
scatter(spaceHFO_svm(:,1),spaceHFO_svm(:,2),'.','c')
scatter(trainArtN(:,1),trainArtN(:,2),'r')
scatter(trainHFON(:,1),trainHFON(:,2),'b')
title('SVM')
hold off

%% 
% <latex>
%  \item In a few sentences, report some observations about the three
%  plots, especially similarities and differences between them. Which
%  of these has overfit the data the most? Which has underfit the data
%  the most? (3 pts)
% </latex>

%%

% They all seem to share a basic diagonal barrier in the center, separating
% the main clusters of artifacts and HFOs. However, the logistical
% regression seems underfit, as it merely produces a straight line, whereas
% the knn plot seems overfit with many small, isolated groups appearing all
% over the plot. The SVM plot seems to have a fit in between the other two,
% with a few large groups that conform to most, but not all, of the data.

%% 
% <latex>
% \end{enumerate}
% \section{Cross-Validation (17 pts)} In this section, you will
% investigate the importance of cross-validation, which is essential
% for choosing the tunable parameters of a model (as opposed to the
% internal parameters the the classifier ``learns'' by itself on the
% training data). 
% \begin{enumerate}
%  \item Since you cannot do any validation on the testing set, you'll
%  have to split up the training set. One way of doing this is to
%  randomly split it into $k$ unique ``folds,'' with roughly the same
%  number of samples ($n/k$ for $n$ total training samples) in each
%  fold, training on $k-1$ of the folds and doing predictions on the
%  remaining one.
%  In this section, you will do 10-fold cross-validation, so create a
%  cell array\footnote{A cell array is slightly different from a
%  normal Matlab numeric array in that it can hold elements of
%  variable size (and type), for example \texttt{folds\{1\} = [1 3 6]; folds\{2\}
%  = [2 5]; folds\{3\} = [4 7];}.} \texttt{folds} that contains 10
%  elements, each of which is itself an array of the indices of
%  training samples in that fold. You may find the \texttt{randperm}
%  function useful for this.
%  Using the command \texttt{length(unique([folds\{:\}]))}, show that
%  you have 200 unique sample indices (i.e. there are no repeats
%  between folds). (2 pts)
% </latex>

%%
% 

randInd = randperm(length(type));
rand_normFeats_train = normFeats_train(randInd,:);

k = 10;
folds = cell(1,k);
mat = zeros(numel(type)/k,1);
for qq = 1:10
    for pp = 1:20
        mat(pp,:) = randInd((qq-1)*20+pp);
    end
    folds{qq} = mat;
end

length(unique([folds{:}]))


%% 
% <latex>
%  \item Train a new $k$-NN model (still using the default parameters)
%  on the folds you just created. Predict the labels for each fold
%  using a classifier trained on all the other folds. After running
%  through all the folds, you will have label predictions for each
%  training sample.
%   \begin{enumerate}
%    \item Compute the error (called the validation error) of these
%    predictions. (3 pts) \item How does this error compare (lower,
%    higher, the same?) to the error you found in question 2.3? Does
%    it make sense? (2 pts)
% </latex>

%%
% 

rand_type = type(randInd);
rand_type = reshape(rand_type,numel(rand_type),1);
fold_class = NaN*ones(numel(mat),length(folds));
for hh = 1:length(folds)
    
    temp_train = rand_normFeats_train;
    temp_train(numel(mat)*(hh-1)+1:numel(mat)*hh,:) = [];
    temp_type = rand_type;
    temp_type(numel(mat)*(hh-1)+1:numel(mat)*hh) = [];
    temp_sample = normFeats_train(cell2mat(folds(hh)),:);
    
    fold_class(:,hh) = knnclassify(temp_sample, temp_train , temp_type);

end
fold_class = reshape(fold_class,numel(rand_type),1);

fold_error = 100*sum(abs(fold_class - rand_type))/numel(rand_type)

% The error here is a bit higher, which makes sense given that each sample
% is being trained on a smaller number of data points.
%% 
% <latex>
%   \end{enumerate}
%  \item Create a parameter space for your $k$-NN model by setting a
%  vector of possible $k$ values from 1 to 30. For each values of $k$,
%  calculate the validation error and average training error over the
%  10 folds.
%   \begin{enumerate}
%    \item Plot the training and validation error values over the
%    values of $k$, using the formatting string \texttt{'b-o'} for the
%    validation error and \texttt{'r-o'} for the training error. (4
%    pts) \item What is the optimal $k$ value and its error? (1 pts)
%    \item Explain why $k$-NN generally overfits less with higher
%    values of $k$. (2 pts)
% </latex>

%%
% 


rand_type = type(randInd);
rand_type = reshape(rand_type,numel(rand_type),1);

train_class =  NaN*ones(1,numel(type));
train_error = NaN*ones(1,30);
fold_error = NaN*ones(1,30);
for kk = 1:30
fold_class = NaN*ones(numel(mat),length(folds));
    for hh = 1:length(folds)
    
    temp_train = rand_normFeats_train;
    temp_train(numel(mat)*(hh-1)+1:numel(mat)*hh,:) = [];
    temp_type = rand_type;
    temp_type(numel(mat)*(hh-1)+1:numel(mat)*hh) = [];
    temp_sample = normFeats_train(cell2mat(folds(hh)),:);
    
    fold_class(:,hh) = knnclassify(temp_sample, temp_train , temp_type, kk);
    end
    
fold_class = reshape(fold_class,numel(rand_type),1);
fold_error(kk) = 100*sum(abs(fold_class - rand_type))/numel(rand_type);

train_class = knnclassify(rand_normFeats_train, rand_normFeats_train , rand_type,kk);
train_class = reshape(train_class,numel(rand_type),1);
train_error(kk) = 100*sum(abs(train_class - rand_type))/numel(rand_type);
end

figure(8)
hold on
plot(train_error,'r-o')
plot(fold_error,'b-o')
xlabel('k','FontSize',12)
legend('Training Error', 'Validation Error')
hold off


% Optimal k value appears to be around 11. It's error is:
optimal_tError = train_error(11)
optimal_vError = fold_error(11)

% Higher values of k create smoother classifier boundaries, with most 
% random fluctions being averaged out. Likewise, low
% values of k attribute too much significance to random fluctuations in a
% particular data set, making them susceptible to overfitting.
%% 
% <latex>
%   \end{enumerate}
%  \item
%   \begin{enumerate}
%    \item Using your optimal value for $k$ from CV, calculate the
%    $k$-NN model's \textit{testing} error. (1 pts) \item How does this
%    model's testing error compare to the $k$-NN model
%    you trained in question 3.3? Is it the best of the three models
%    you trained in Section 3? (2 pts)
% </latex>

%%

knn_optimal = knnclassify(normFeats_test, normFeats_train, type,11);

kOptimalError = 100*sum(abs(knn_optimal - type_test'))/numel(type_test)

% This error (11.90%) is lower than the error in 3.3. (17.38%). However, it
% is still not lower than the svm error (11.19%), but is lower than the
% logistical error (13.57%).
%% 
% <latex>
%   \end{enumerate}
% \end{enumerate}
% </latex>
