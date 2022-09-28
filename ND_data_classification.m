%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%设置两组数据并使用感知机将其分类
%两组数据的均值分别为0和5
%2022.9.27
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
rng(1);%让后面产生的正态分布随机数保持不变，以便于分析

m1=0;%第一组数据的均值
m2=5;%第二组数据的均值
N=10;%每组数据的数量
d=2;%数据的维度

M1=[m1 m1];
M2=[m2 m2];
sigma=[1 0.5; 0.5 2];
R=chol(sigma);
z1=repmat(M1,N,1)+randn(N,d)*R;%第一组数据
z2=repmat(M2,N,1)+randn(N,d)*R;%第二组数据
x1=z1(:,1);
y1=z1(:,2);
x2=z2(:,1);
y2=z2(:,2);
scatter(x1,y1,'r','o');
hold on
scatter(x2,y2,'b','+');
hold on
w=ones(1,2);
b=0;
X=[x1;x2];
Y=[y1;y2];
F=[X,Y];
eta=0.1;
% while(1)
for j=1:200
    for i=1:N*2
%         delta_w=(t(i)-step(w*F(i,:)'+b));
        delta_b=(t(i)-step(w*F(i,:)'+b));
%         w=w+(delta_w)'*eta;
        b=b+delta_b*eta;
%         W(i,:)=w;
        B(i,:)=b;
%         delta_w_and_b(1:2,i)=delta_w;
%         delta_w_and_b(3,i)=delta_b;
    end
%        if  delta_w_and_b==zeros(3,length(delta_w_and_b))
%            break
%        end

end
% W1=W(length(W),1);
% W2=W(length(W),2);
X=-10:1:10;
% Y=-1*W1/W2*X+B(length(B),1)/W2;
Y=-1*X-B(length(B),1);
plot(X,Y)
     

function t=t(i)
    if i<=10
        t=0;
    else
        t=1;
    end
end


%创建阶跃函数step
function step=step(x)
    if x<0
        step=0;
    else
        step=1;
    end
end












