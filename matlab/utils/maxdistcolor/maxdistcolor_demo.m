%%% MAXDISTCOLOR Demo Script %%%
% Plot the colormap in the UCS, as a distance plot, and as colorbars with colornames.
%
% The CAM02UCS colorspace functions must be obtained separately, for
% example my implementation here: <https://github.com/DrosteEffect/CIECAM02>
%
%%% Define the colorspace function:
fun = @sRGB_to_OKLab;   % OKLab (an improvement over CIELab)
%fun = @sRGB_to_CIELab; % CIELab (not particularly uniform colorspace)
%fun = @(m)sRGB_to_CAM02UCS(m,true,'LCD');      % CAM02-LCD (recommended)
%fun = @(m)sRGB_to_CAM02UCS(m,true,'UCS');      % CAM02-UCS
%fun = @(m)sRGB_to_OSAUCS(m,true,true);         % OSA-UCS (modified)
%fun = @(m)CIELab_to_DIN99o(sRGB_to_CIELab(m)); % DIN99o
%fun = @(m)CIELab_to_DIN99(sRGB_to_CIELab(m));  % DIN99
%
%%% Generate colormap of distinctive colors:
[rgb,ucs,sts] = maxdistcolor(9,fun);
%[rgb,ucs,sts] = maxdistcolor(9,fun,'exc',[]);
%[rgb,ucs,sts] = maxdistcolor(9,fun,'class','single');
%[rgb,ucs,sts] = maxdistcolor(9,fun,'Cmin',0.5,'Cmax',0.6);
%[rgb,ucs,sts] = maxdistcolor(9,fun,'Lmin',0.4,'Lmax',0.6);
%[rgb,ucs,sts] = maxdistcolor(9,fun,'inc',[0,0,0;1,0,1],'exc',[0,1,0]);
%[rgb,ucs,sts] = maxdistcolor(9,fun,'sort','longest','disp','verbose');
%[rgb,ucs,sts] = maxdistcolor(9,fun, 'bitR',8,'bitG',8,'bitB',8); % Truecolor -> slow!
%[rgb,ucs,sts] = maxdistcolor(64,fun,'bitR',2,'bitG',2,'bitB',2, 'exc',[]); % entire RGB gamut.
N = size(rgb,1);
%
%%% Plot color distance matrix:
figure();
for k = 1:N
	dst = sqrt(sum(bsxfun(@minus,ucs,ucs(k,:)).^2,2));
	scatter3(k*ones(1,N),1:N,dst, 123, rgb,...
		'MarkerFaceColor',rgb(k,:), 'LineWidth',2.8, 'Marker','o')
	hold on
end
title(sprintf('Colormap Euclidean Distances in %s Colorspace',sts.colorspace))
zlabel('Euclidean Distance')
ylabel('Colormap Index')
xlabel('Colormap Index')
set(gca,'XTick',1:N,'YTick',1:N)
%
%%% Plot colors in UCS:
figure();
scatter3(ucs(:,3),ucs(:,2),ucs(:,1), 256, rgb, 'filled')
text(ucs(:,3),ucs(:,2),ucs(:,1),cellstr(num2str((1:N).')), 'HorizontalAlignment','center')
%
%%% Plot outline of RGB cube:
M = 23;
[X,Y,Z] = ndgrid(linspace(0,1,M),0:1,0:1);
mat = fun([X(:),Y(:),Z(:);Y(:),Z(:),X(:);Z(:),X(:),Y(:)]);
X = reshape(mat(:,3),M,[]);
Y = reshape(mat(:,2),M,[]);
Z = reshape(mat(:,1),M,[]);
line(X,Y,Z,'Color','k')
axis('equal')
title(sprintf('Colormap in %s Colorspace',sts.colorspace))
zlabel(sts.axesLabels{1})
ylabel(sts.axesLabels{2})
xlabel(sts.axesLabels{3})
%
%%% Plot colorband image:
figure()
image(permute(rgb,[1,3,2]))
title('Colormap in Colorbands')
ylabel('Colormap Index')
set(gca,'XTick',[], 'YTick',1:N, 'YDir','normal')
%%% Add colornames (if COLORNAMES is available):
try %#ok<TRYNC>
	text(ones(1,N), 1:N, colornames('CSS',rgb),...
		'HorizontalAlignment','center', 'BackgroundColor','white')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%maxdistcolor_demo