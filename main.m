close all
clear all
clc

data = csvread('train.csv',1);
inds = data(:,1);
y = data(:,2);
X = data(:,3:end);

hist(X(:,16));