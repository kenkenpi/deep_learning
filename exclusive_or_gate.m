%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%2输入，单输出的异或非线性分类问题
%隐藏层和输出层都采用sigmoid函数
%2022.12.10~13
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;clear;clc;
%% inicialization

X=[0 0;
   1 0;
   0 1;
   1 1];
T=[0 1 1 0];

N=300;
w=[0.1 0.1;
   0.1 0.1];
v=[0.1;-0.1];
b=[0;0];
c=0;
eta=0.1;
sum_w=zeros(2,2);
sum_v=zeros(2,1);
sum_b=zeros(2,1);
sum_c=0;

%% trainning
for i=1:N
    for j=1:4
        p=w*(X(j,:))'+b;
        p1=p(1,1);
        p2=p(2,1);
        fp(1,1)=sigmoid(p1);
        fp(1,2)=sigmoid(p2);
        q=v(1,1)*fp(1,1)+v(2,1)*fp(1,2)+c;
        y=sigmoid(q);
        delta_v=(T(j)-y)*fp';
        delta_c=(T(j)-y);
        dq_dp11=sigmoid(p1)*(1-sigmoid(p1));
        dq_dp12=0;
        dq_dp21=0;
        dq_dp22=sigmoid(p2)*(1-sigmoid(p2));
        dq_dp=[dq_dp11 dq_dp12;dq_dp21 dq_dp22];
        delta_w=delta_c*dq_dp*v*X(j,:);
        delta_b=delta_c*dq_dp*v;
        sum_w=sum_w+delta_w;
        sum_b=sum_b+delta_b;
        sum_v=sum_v+delta_v;
        sum_c=sum_c+delta_c;
    end
    w=w+sum_w*eta;
    v=v+sum_v*eta;
    b=b+sum_b*eta;
    c=c+sum_c*eta;
end

%% test

for k=1:4
        p=w*(X(k,:))'+b;                                                
        p1=p(1,1);
        p2=p(2,1);
        fp(1,1)=sigmoid(p1);
        fp(1,2)=sigmoid(p2);
        q=v(1,1)*fp(1,1)+v(2,1)*fp(1,2)+c;
        Y(k)=sigmoid(q);
end
Y
%% function

function sigmoid=sigmoid(x)
    sigmoid=1/(1+exp(-1*x));
end
%%












