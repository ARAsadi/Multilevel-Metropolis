d=load('mnist.mat');
global train_X_double
global trainY_one_hot
global test_X_double
global testY_one_hot

train_X=trainX';
train_X_double=double(train_X);
train_X_double=train_X_double/256; % Normalizing to [0,1]

test_X=testX';
test_X_double=double(test_X);
test_X_double=test_X_double/256; % Normalizing to [0,1]

%One-hot the labels:
trainY_one_hot=zeros(10,60000);
for i=1:60000
    trainY_one_hot(:,i)=bsxfun(@eq, transpose(0:9), d.trainY(1,i));
end

%One-hot the labels:
testY_one_hot=zeros(10,10000);
for i=1:10000
    testY_one_hot(:,i)=bsxfun(@eq, transpose(0:9), d.testY(1,i));
end