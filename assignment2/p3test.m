%% Neural Network with one hidden layer
% Neural network for handwritten digit classification
% Model of neural network - 64-70-2

clear;
clc;
close all;

%% Training Phase
tic

filename = 'train.txt';
X = dlmread(filename, ',');
n = size(X, 1); % Number of training samples
t = X(:, size(X, 2));
%X = X(:, 1 : (size(X, 2) - 1));
X(:, size(X, 2)) = 1;
X = X';

% Initialization by feedforward operation
%bias1 = 1;
%bias2 = 1;
nHiddenUnits = 70;
f = @(x) logsig(x);
df = @(x) logsig(x) .* (1 - logsig(x));

a = -1/sqrt(size(X, 1));
b = 1/sqrt(size(X, 1));
wji = (b - a) .* rand(nHiddenUnits, size(X, 1)) + a;
netj = wji * X;

% Apply fnet from 1 : 32
yj(1 : nHiddenUnits - 1, :) = f(netj(1 : nHiddenUnits - 1, :));
yj(nHiddenUnits, :) = 1;
a = -1/sqrt(size(netj, 1));
b = 1/sqrt(size(netj, 1));
wkj = (b - a) .* rand(2, size(yj, 1)) + a;
netk = wkj * yj;
zk = f(netk);

% Backpropagation operation
tk(:, 1) = t < 6;
tk(:, 2) = t > 6;
tk = tk';
Jw = 0.5 * sum((tk - zk) .^ 2); % Gives error for each training sample in each column
delk = (tk - zk) .* df(netk);
delj = df(netj) .* (wkj' * delk);

eta = 1;
k = 0;
theta = 0.7;
while k < n
    k = mod(k, n) + 1; %Kth training sample
    xk = X(:, k);
    netj(:, k) = wji * X(:, k);
    yj(1 : nHiddenUnits - 1, k) = f(netj(1 : nHiddenUnits - 1, k));
    yj(nHiddenUnits, k) = 1;
    netk(:, k) = wkj * yj(:, k);
    zk(:, k) = f(netk(:, k));
    delk(:, k) = (tk(:, k) - zk(:, k)) .* df(netk(:, k));
    % delj(:, k) = df(netj(:, k)) * delk(:, k)' * wkj(:, k);
    delj(:, k) = df(netj(:, k)) .* (wkj' * delk(:, k));
    wkj = wkj + eta * delk(:, k) * yj(:, k)';
    wji = wji + eta * delj(:, k) * xk';
    
    % Updating all values according to new weights
    
    netj = wji * X;
    yj(1 : nHiddenUnits - 1, :) = f(netj(1 : nHiddenUnits - 1, :));
    yj(nHiddenUnits, :) = 1;
    netk = wkj * yj;
    zk = f(netk);
    
    % Error
    Jw(k) = 0.5 * sum((tk(:, k) - zk(:, k)) .^ 2);
end


%% Testing Phase


clear tk zk
filename = 'test.txt';
X = dlmread(filename, ',');
n = size(X, 1); % Number of test samples
t = X(:, size(X, 2));
% X = X(:, 1 : 64);
X(:, size(X, 2)) = 1;
X = X';
zk = zeros(2, size(X, 2));
tk(:, 1) = t < 6;
tk(:, 2) = t > 6;
tk = tk';
k = 0;

while k < n
    k = mod(k, n) + 1;
    xk = X(:, k);
    netj(:, k) = wji * X(:, k);
    yj(1 : nHiddenUnits - 1, k) = f(netj(1 : nHiddenUnits - 1, k));
    yj(nHiddenUnits, k) = 1;
    netk(:, k) = wkj * yj(:, k);
    zk(:, k) = f(netk(:, k));
end

zk = zk > 0.5;
err = tk - zk;
numberofErrors = sum(max(err ~= 0))
accuracy = (size(X, 2) - (numberofErrors))/size(X, 2)

toc