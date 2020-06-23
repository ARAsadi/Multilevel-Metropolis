%{THIS SOURCE CODE IS SUPPLIED “AS IS” WITHOUT WARRANTY OF ANY KIND, AND ITS AUTHOR AND THE JOURNAL OF
MACHINE LEARNING RESEARCH (JMLR) AND JMLR’S PUBLISHERS AND DISTRIBUTORS, DISCLAIM ANY AND ALL WARRANTIES,
INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, 
AND ANY WARRANTIES OR NON INFRINGEMENT. THE USER ASSUMES ALL LIABILITY AND RESPONSIBILITY FOR USE OF THIS
SOURCE CODE, AND NEITHER THE AUTHOR NOR JMLR, NOR JMLR’S PUBLISHERS AND DISTRIBUTORS, WILL BE LIABLE FOR
DAMAGES OF ANY KIND RESULTING FROM ITS USE. Without limiting the generality of the foregoing, neither the author,
nor JMLR, nor JMLR’s publishers and distributors, warrant that the Source Code will be error-free, will operate 
without interruption, or will meet the needs of the user.%}

rng('default') % For reproducibility
%% Parameters
global train_X_double
global trainY_one_hot % Labels in one-hot format
global test_X_double
global testY_one_hot % Labels in one-hot format
global M_1 % Reference matrix
global M_2 % Reference matrix

input_dim=784; % Input dimension
hidden=100; % Size of hidden layer

M_1=eye(hidden,input_dim);
M_2=eye(10,hidden);
T=40000;
T_2=10; %Inner level running time
s_prop_1=0.001; % Variance of proposal distribution q_1
s_prop_2=0.0005; % Variance of proposal distribution q_2


%Temperature vector:
a_1=2*10^(-6);
a_2=10^(-6);

%% Histories 
Hist_train=zeros(1,T); % Training errors
Hist_test=zeros(1,T); % Test errors

%% Initialization
W_1=zeros(hidden,input_dim);
W_2=zeros(10,hidden);

%% The algorithm

for t=1:T
    t % Output the iteration
    W_1_hat=W_1+normrnd(0,s_prop_1,[hidden,input_dim]); % Gaussian proposal q_1
    num_A=0; % This is the numerator of A.
    for i=1:T_2
        W_2_hat=W_2+normrnd(0,s_prop_2,[10,hidden]); % Gaussian proposal q_2
        B=(f(W_1,W_2_hat)/f(W_1,W_2))^(1/a_2);
        u=rand; 
        if (u<B)
            W_2=W_2_hat;
            disp('W_2 Changed')
        end
            num_A=num_A+(f(W_1_hat,W_2)/f(W_1,W_2))^(1/a_2);
    end
    norm_W_1=norm(W_1)
    norm_W_2=norm(W_2)
    
    train_error=Neural_Net_Loss(train_X_double, trainY_one_hot,W_1+M_1,W_2+M_2)
    test_error=Neural_Net_Loss(test_X_double, testY_one_hot,W_1+M_1,W_2+M_2)
    Hist_train(t)=train_error;
    Hist_test(t)=test_error;
    
    A=num_A/(T_2);
    C=A^(a_2/(a_1+a_2));
    u=rand;
    if (u<C)
        W_1=W_1_hat;
        disp('W_1 Changed')
    end
end
figure
plot(1:T,Hist_train,1:T,Hist_test)

%% Functions
function [Loss] = f(W_1,W_2)
    global train_X_double
    global trainY_one_hot
    global M_1
    global M_2
    Loss=exp(-Neural_Net_Loss(train_X_double, trainY_one_hot,W_1+M_1,W_2+M_2));
end

function [Loss]=Neural_Net_Loss(input_instance, input_label,V_1,V_2)
x_1 = max(V_1*input_instance, zeros(size(V_1*input_instance))); % The max is for ReLU.
x_2 = V_2*(x_1);

% Softmax output:
    Net_output=x_2;
    for i=1:max(size(Net_output))
        for j=1:min(size(Net_output))
        Net_output(j,i)=exp(x_2(j,i))/(sum(exp(x_2(:,i))));
        end
    end
    
Loss_output=sum((input_label- Net_output).^2); % Squared l_2 loss
Loss=mean(Loss_output); % Average
end
