
test = test2;
y = y2;


DD = size(test,1);

ave = sum(test,2)/(claTestNum + edmTestNum);

tic
S = 1/(claTestNum + edmTestNum - 1).*(test - ave*ones(1,claTestNum + edmTestNum))*(test - ave*ones(1,claTestNum + edmTestNum)).';
toc

tic
[V,D] = eig(S);
[~,I] = sort(diag(D),'descend');
toc

clear S

%%


coords = (V(:,I([1,2]))')*test;

figure
plot(coords(1,testclasses==1),coords(2,testclasses==1),'.r')
hold on
plot(coords(1,testclasses==-1),coords(2,testclasses==-1),'.b')
legend('Classical MIDI files','EDM MIDI files','Location','southwest')
hold off
title('Test Midi files colored by true class')

figure
pointsize = 10;
scatter(coords(1,:),coords(2,:),pointsize,y)
colorbar
title('Test MIDI files colored by SVM output value')
h = colorbar;
ylabel(h, 'SVM output value')




