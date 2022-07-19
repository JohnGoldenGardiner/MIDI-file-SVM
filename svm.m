



%% Constants

EDM_DIR = ['C:\Users\JohnGardiner\Documents\Classes\' ...
           'MATH 156, Machine Learning\machine learning project\' ...
           'midi_files\edm']; % Directory where your edm MIDI files are

CLASSICAL_DIR = ['C:\Users\JohnGardiner\Documents\Classes\' ...
                 'MATH 156, Machine Learning\machine learning project\' ...
                 'midi_files\classical']; 
                        % Directory where your classical MIDI files are

edmTestNum = 200; % number of EDM test datapoints
claTestNum = 200; % number of classical test datapoints

edmTrainingNum = 4500; % number of EDM training datapoints
claTrainingNum = 4500; % number of classical training datapoints

D = 400; % Number of notes sampled from each piece
c = 12; % Number of transformed copies of each data point



%% Initialize

% Change path to include midi reading library.
addpath(...
    'C:\Users\JohnGardiner\Documents\Classes\MATH 156, Machine Learning\machine learning project\midi_lib',...
    'C:\Users\JohnGardiner\Documents\Classes\MATH 156, Machine Learning\machine learning project\matlab-midi-master\src'...
    )

% initialize variables
index = [1]; % Keeps track of where pieces begin and end list of notes
noteMatrices = zeros(50000000,4); % Rows are notes. Make this big enough



%% Read EDM MIDI files.

edmFiles = dir(EDM_DIR); % List contents of directory
edmFiles([edmFiles.isdir])=[]; % Keep only files in the directory
edmFileNames = {edmFiles.name}; % Get names of files
edmFileNames = edmFileNames(cellfun(@(x) strcmp(x(end-3:end),'.mid'), ...
                                    edmFileNames)); % Keep only MIDI files
edmNum = length(edmFileNames); % Number of edm MIDI files.

% Read MIDI files and story in noteMatrices
for k=1:edmNum
    notes = midiInfo(readmidi([EDM_DIR,'\',edmFileNames{k}]),0);
    index(k+1,1) = size(notes,1) + index(k,1);
    notes = notes(:,[5,6,3,4]);
    noteMatrices(index(k):(index(k+1)-1),:) = notes;
    if mod(k,50)==0
        fprintf('%d MIDI files read and saved.\n',k)
    end
end
fprintf('%d EDM MIDI files read and saved.\n',k)



%% Read classical MIDI files.

claFiles = dir(CLASSICAL_DIR); % List contents of directory.
claFiles([claFiles.isdir])=[]; % Keep only files.
claFileNames = {claFiles.name};
claFileNames = claFileNames(cellfun(@(x) strcmp(x(end-3:end),'.mid'), ...
                                    claFileNames)); % Keep only MIDI files
claNum = length(claFileNames); % Number of classical MIDI files.

for k=1:claNum
    notes = midiInfo(readmidi([CLASSICAL_DIR,'\',claFileNames{k}]),0);
    index(k+edmNum+1,1) = size(notes,1) + index(k+edmNum,1);
    notes = notes(:,[5,6,3,4]);
    noteMatrices(index(k+edmNum):(index(k+edmNum+1)-1),:) = notes;
    if mod(k,50)==0
        fprintf('%d MIDI files read and saved.\n',k)
    end
end
fprintf('%d Classical MIDI files read and saved.\n',k)



%% Useful variables and cleanup

totNum = edmNum + claNum; % Total number of MIDI files.
noteNum = index(end) - 1; % Total number of notes in all MIDI files.
classes = [ones(edmNum,1);2*ones(claNum,1)]; % Genre classifications

noteMatrices = noteMatrices(1:noteNum,:);



%% Process first kernel data

% Each column of K1data is a data point, a vector of dimension D*4.
K1data = zeros(D*4,totNum*c); % Initialize

for k=1:totNum

    notes = noteMatrices(index(k):index(k+1)-1,:);
    
    for i=1:c
    
        if index(k+1)-index(k)>D
            newNotes = notes(randperm(size(notes,1),D),:);
        else
            copies = floor(D./size(notes,1));
            augnotes = [repmat(notes,[copies,1]);notes(randperm(size(notes,1),D-copies*size(notes,1)),:)];
            newNotes = augnotes(randperm(D),:);
        end

        pitchshift = i-6;
        minpitch = min(newNotes(:,3));
        maxpitch = max(newNotes(:,3));
        newNotes(:,3) = newNotes(:,3) + min([127-maxpitch,max([pitchshift,-minpitch])]);
        
        newNotes = reshape(newNotes,[D*4,1]);
        
        K1data(:,(k-1)*c+i) = newNotes;

    end
end

K1classes = kron(classes,ones(c,1));



%% Process second kernel data (chord representation)

% Keeps track of where different pieces begin and end in the list of
% chords.
chordindex = [1]; % Initialized here

chordNum = 0;
for k=1:totNum
    notes = noteMatrices(index(k):index(k+1)-1,:);
    notes = notes(:,1:3); % Remove volume data.
    notes(:,3) = mod(notes(:,3),12); % Make pitches modulo 12.
    times = unique(notes(:,1:2));
    chordNum = chordNum + length(times);
end

% chords is a list of the chords in the order they occur.
chords = zeros(chordNum,12); % Initialized here

for k=1:totNum
    
    notes = noteMatrices(index(k):index(k+1)-1,:);
    notes = notes(:,1:3); % Remove volume data.
    notes(:,3) = mod(notes(:,3),12); % Make pitches modulo 12.

    [times,~,timesind] = unique(notes(:,1:2)); % Find unique times
    T = 1:length(times); % Index for the unique times in the piece
    events = reshape(T(timesind),size(notes,1),2);
        % First column is begin time indices and second column is end time
        % indices.
    
    % Collect all notes playing between two consecutive events.
    for i=T
        % Find indices of notes whose beginning is is before or at i and
        % whose ending is after i.
        logic = events(:,1)<=i & events(:,2)>i;
        pitches = unique(notes(logic,3)) + 1; % The corresponding pitches
        chords(chordindex(k) - 1 + i,pitches) = 1;
    end

    chordindex(k+1) = chordindex(k) + length(times);

end


% Each column of K2data is a data point, a vector of dimension D*12.
K2data = zeros(D*12,totNum*c); % Initialize

for k=1:totNum
    
    notes = chords(chordindex(k):chordindex(k+1)-1,:);
    %notes = notes(1:end-4,:) + notes(2:end-3,:) + notes(3:end-2,:) + notes(4:end-1,:) + notes(5:end,:);

    for i=1:c
        
        if size(notes,1) > D
            newnotes = notes(randperm(size(notes,1),D),:);
        else
            copies = floor(D./size(notes,1));
            augnotes = [repmat(notes,[copies,1]);notes(randperm(size(notes,1),D-copies*size(notes,1)),:)];
            newnotes = augnotes(randperm(D),:);
        end

        pitchshift = i;
        newnotes = circshift(newnotes,[0,i]);
        
        newnotes = reshape(newnotes,[D*12,1]);
        
        K2data(:,(k-1)*c+i) = newnotes;
        
    end

end


K2classes = kron(classes,ones(c,1));



%% Prepare test data

testEdmIndex = randsample(edmNum,edmTestNum);
testClaIndex = randsample(claNum,claTestNum);
testIndex = c*([testEdmIndex;edmNum + testClaIndex] - 1)...
            + randi(c,edmTestNum+claTestNum,1);

test1 = K1data(:,testIndex); % Test datapoints, for use with first kernel
test2 = K2data(:,testIndex); % Test datapoints, for use with second kernel



%% Prepare training data

eligibleDatapoints = ones(1,totNum);
% Avoid test datapoints
eligibleDatapoints((testIndex - mod(testIndex,c))./c + 1) = 0;
eligibleDatapoints = kron(eligibleDatapoints,ones(1,c));

edmEligible = eligibleDatapoints;
claEligible = eligibleDatapoints;
edmEligible(edmNum*c+1:end) = 0;
claEligible(1:edmNum*c) = 0;

r = 1:totNum*c;

edmEligibleIndices = r(logical(edmEligible));
claEligibleIndices = r(logical(claEligible));

trainingEdmIndex = randsample(edmEligibleIndices,edmTrainingNum);
trainingClaIndex = randsample(claEligibleIndices,claTrainingNum);

% First kernel training data
training1 = [K1data(:,trainingEdmIndex),K1data(:,trainingClaIndex)];
% Second kernel training data
training2 = [K2data(:,trainingEdmIndex),K2data(:,trainingClaIndex)]; 

% The true classification targets, -1 for EDM, +1 for classical
targets = [-ones(1,edmTrainingNum),ones(1,claTrainingNum)].';




%% Construct training kernels

% Scalings are the distance to the kk nearest training neighbor.
kk1 = 10;
sortedDistances1 = sort(diffs(training1,training1));
kk1Distance = sortedDistances1(kk1+1,:);
trainingScalings1 = (kk1Distance'*kk1Distance).^(1/2);

% The first kernel
K1 = exp(-diffs(training1,training1)./trainingScalings1);

% Scalings are the distance to the kk nearest training neighbor
kk2 = 10;
sortedDistances2 = sort(diffs(training2,training2));
kk2Distance = sortedDistances2(kk2+1,:);
trainingScalings2 = (kk2Distance'*kk2Distance).^(1/2);

% The second kernel (data represented as chords)
K2 = exp(-diffs(training2,training2)./trainingScalings2);



%% Construct test kernels

% Kernel values between training and test data

sortedTestDistances1 = sort(diffs(training1,test1));
kk1TestDistance = sortedTestDistances1(kk1,:);
testScalings1 = (kk1TestDistance'*kk1Distance).^(1/2);

k1 = exp(-diffs(test1,training1)./testScalings1);

sortedTestDistances2 = sort(diffs(training2,test2));
kk2TestDistance = sortedTestDistances2(kk2,:);
testScalings2 = (kk2TestDistance'*kk2Distance).^(1/2);

k2 = exp(-diffs(test2,training2)./testScalings2);



%% Set up and solve the quadratic programming problem

C = 30; % Try varying this.

H1 = diag(targets)*K1*diag(targets);
H2 = diag(targets)*K2*diag(targets);

f = -ones(edmTrainingNum+claTrainingNum,1);

Aeq = targets.';
beq = 0;

lb = zeros(edmTrainingNum+claTrainingNum,1);
lu = C*ones(edmTrainingNum+claTrainingNum,1);

options = optimset('Algorithm','interior-point-convex');

fprintf('quadprog started for kernel 1.\n')
tic
a1 = quadprog(H1,f,[],[],Aeq,beq,lb,lu,[],options);
toc

fprintf('quadprog started for kernel 2.\n')
tic
a2 = quadprog(H2,f,[],[],Aeq,beq,lb,lu,[],options);
toc

small = 10^-6;
vm1 = (a1 > small)&(a1 < (C-small));
vs1 = a1 > small;
Nm1 = sum(vm1);
Ns1 = sum(vs1);
vm2 = (a2 > small)&(a2 < (C-small));
vs2 = a2 > small;
Nm2 = sum(vm2);
Ns2 = sum(vs2);

b1 = 1/Nm1*sum(vm1.*(targets-K1*(vs1.*targets.*a1)));
b2 = 1/Nm2*sum(vm2.*(targets-K2*(vs2.*targets.*a2)));



%% Accuracy on test data

y1 = k1*(vs1.*targets.*a1) + b1; % SVM output values for first kernel
y2 = k2*(vs2.*targets.*a2) + b2; % SVM output values for second kernel

testclasses = [-ones(1,edmTestNum),ones(1,claTestNum)].'; % True classes

predictions1 = sign(y1); % SVM classification for first kernel
accuracy1 = sum(testclasses==predictions1)/length(y1); % Percent correct

predictions2 = sign(y2); % SVM classification for second kernel
accuracy2 = sum(testclasses==predictions2)/length(y2);

