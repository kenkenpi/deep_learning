%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%2输入、3输出的三分类逻辑回归
%逻辑回归：概率分类模型(激活函数：softmax)
%每个分类都生成遵循平均值μ≠0的正态分布的样本数据
%每个分类都有100个数据，即对300个数据进行分类
%2022.9.30
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear;
clear all;
%rng(1);%让后面产生的正态分布随机数保持不变，以便于分析
%%

m11=0;
m12=10;%第一组数据的均值
m21=5;
m22=5;%第二组数据的均值
m31=10;
m32=0;%第三组数据的均值
N=100;%每组数据的数量
d=2;%数据的维度
K=3;%分了K类
M=N*K;%全部数据的个数

M1=[m11 m12];
M2=[m21 m22];
M3=[m31 m32];
sigma=[2 0.5; 0.5 4];
R=chol(sigma);
z1=repmat(M1,N,1)+randn(N,d);%第一组数据
z2=repmat(M2,N,1)+randn(N,d);%第二组数据
z3=repmat(M3,N,1)+randn(N,d);%第三组数据
x1=z1(:,1);
y1=z1(:,2);
x2=z2(:,1);
y2=z2(:,2);
x3=z3(:,1);
y3=z3(:,2);

scatter(x1,y1,'r','o');
hold on
scatter(x2,y2,'b','+');
hold on
scatter(x3,y3,'g','p');
hold on
%%
w11=0.1;
w12=1;
w21=0.1;
w22=1;
w1=[w11 w12];
w2=[w21 w22];
w=[w1;w2];
b=[0;0];
X=[x1;x2;x3];
Y=[y1;y2;y3];
F=[X,Y];
eta=0.1; 
delta_w_sum=zeros(2,2);
for j=30000
    for i=1:N*K
        [cerrect_o1,cerrect_o2]=t(i);
        T(:,i)=[cerrect_o1,cerrect_o2]';
        y(:,i)=w*F(i,:)'+b;
        [output1,output2]=softmax(w*F(i,:)'+b);
        SF(:,i)=[output1,output2]';
        delta_w=(T(:,i)- SF(:,i))*F(i,:);
        delta_b=T(:,i)-SF(:,i);
        delta_w_sum=delta_w_sum+delta_w;
        delta_b_sum=delta_w_sum+delta_b;
    end
        w=w+delta_w_sum*eta;
        b=b+delta_b_sum*eta;
end
X=-10:1:10;
Y1=-1*w(1,1)/w(1,2)*X-b(1,1)/w(1,2);
Y2=-1*w(2,1)/w(2,2)*X-b(2,1)/w(2,2);
plot(X,Y1,X,Y2);
function [output1,output2]=t(i)
N=100;
    if i<=N
        output1=0;
        output2=0;
    else if i>2*N        
        output1=1;
        output2=1;
        else
        output1=1;
        output2=0;
        end
    end
end
function [output1,output2]=softmax(x)
    output1=exp(x(1,1))/(exp(x(1,1))+exp(x(2,1)));
    output2=exp(x(2,1))/(exp(x(1,1))+exp(x(2,1)));
end
%%





















