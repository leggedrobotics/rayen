close all; clc;clear;
doSetup();

% n=3;
S=[5.1 1.1 3.1;
   1.1 4.5 6.8;
   3.1 6.8 8.9];

% tmp=rand(n,n);
% H=tmp'*tmp + 0.01*eye(n);

H= [0.8 0.6 0.9;
    0.6 1.2 1.1
    0.9 1.1 1.4];

all_changes=eig(-inv(H)*S);

tmp=(max(all_changes)-min(all_changes))/9.0

all_delta=linspace(min(all_changes)-tmp,max(all_changes)+tmp,200);
all_eigenvalues=[];
for delta=all_delta
    [V,D] = eig(delta*H+S);
    all_eigenvalues=[all_eigenvalues sort(diag(D))];
end


figure; hold on;
xline(all_changes(1))
xline(all_changes(2))
xline(all_changes(3))
yline(0.0,'--')

plot(all_delta,all_eigenvalues','LineWidth',2)
legend('','','','','(eig$(\delta H +S)$)$_{\left[0\right]}$','(eig$(\delta H +S)$)$_{\left[1\right]}$','(eig$(\delta H +S)$)$_{\left[2\right]}$','Location','northwest')

xlabel('$\delta$');
ylim([-4,7])
xlim([-26.32, 5.85])

set(gcf,'Position',[536   754   552   262])
export_fig eigenvalues.png -m2.5