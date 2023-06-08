function Lab99o = CIELab_to_DIN99o(Lab)
% Convert a matrix of CIELAB L* a* b* values to DIN99o L99o a99o b99o values (DIN 6176).
%
% (c) 2018-2023 Stephen Cobeldick
%
%%% Syntax:
% Lab99o = CIELab_to_DIN99o(Lab)
%
% <https://de.wikipedia.org/wiki/DIN99-Farbraum>
%
%% Inputs and Outputs
%
%%% Input Argument:
% Lab = Numeric Array, size Nx3 or RxCx3, where the last dimension
%       encodes CIELAB values [L*,a*,b*] in the range 0<=L*<=100.
%
%%% Output Argument:
% Lab99o = Numeric Array, same size as <Lab>, where the last dimension
%          encodes the DIN99o values [L99o,a99o,b99o].
%
% See also SRGB_TO_CIELAB CIELAB_TO_DIN99
% SRGB_TO_CAM02UCS SRGB_TO_OKLAB SRGB_TO_OSAUCS
% MAXDISTCOLOR MAXDISTCOLOR_VIEW MAXDISTCOLOR_DEMO

%% Input Wrangling %%
%
isz = size(Lab);
assert(isnumeric(Lab),...
	'SC:CIELab_to_DIN99o:Lab:NotNumeric',...
	'1st input <Lab> must be numeric.')
assert(isreal(Lab),...
	'SC:CIELab_to_DIN99o:Lab:ComplexValue',...
	'1st input <Lab> cannot be complex.')
assert(isz(end)==3,...
	'SC:CIELab_to_DIN99o:Lab:InvalidSize',...
	'1st input <Lab> last dimension must have size 3 (e.g. Nx3 or RxCx3).')
Lab = reshape(Lab,[],3);
assert(all(Lab(:,1)>=0&Lab(:,1)<=100),...
	'SC:CIELab_to_DIN99o:Lab:OutOfRange',...
	'1st input <Lab> L values must be within the range 0<=L<=100')
%
if ~isfloat(Lab)
	Lab = double(Lab);
end
%
kCH = 1;
kE  = 1;
%
%% Lab2DIN99o %%
%
L99o = 303.67/kE * log(1 + 0.0039*Lab(:,1));
eo =      (Lab(:,2).*cosd(26)+Lab(:,3).*sind(26));
fo = 0.83*(Lab(:,3).*cosd(26)-Lab(:,2).*sind(26));
G = sqrt(eo.^2 + fo.^2);
C99o = log(1 + 0.075*G)./(0.0435*kCH*kE);
h99o = atan2(fo,eo) + 26*pi/180;
a99o = C99o .* cos(h99o);
b99o = C99o .* sin(h99o);
%
Lab99o = reshape([L99o,a99o,b99o],isz);
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%CIELab_to_DIN99o