function [Lab,XYZ] = sRGB_to_CIELab(rgb,test)
% Convert a matrix of sRGB R G B values to CIELAB L* a* b* values.
%
% (c) 2018-2023 Stephen Cobeldick
%
%%% Syntax:
% Lab = sRGB_to_CIELab(rgb)
%
% <https://en.wikipedia.org/wiki/SRGB>
% <https://en.wikipedia.org/wiki/CIELAB_color_space>
%
%% Inputs and Outputs
%
%%% Input Argument:
% rgb = Numeric Array, size Nx3 or RxCx3, where the last dimension
%       encodes sRGB values [R,G,B] in the range 0<=RGB<=1.
%
%%% Output Argument:
% Lab = Numeric Array, same size as <rgb>, where the last dimension
%       encodes CIELAB values [L*,a*,b*] in the range 0<=L*<=100.
%
% See also CIELAB_TO_SRGB CIELAB_TO_DIN99 CIELAB_TO_DIN99O
% SRGB_TO_CAM02UCS SRGB_TO_OKLAB SRGB_TO_OSAUCS
% MAXDISTCOLOR MAXDISTCOLOR_VIEW MAXDISTCOLOR_DEMO

%% Input Wrangling %%
%
isz = size(rgb);
assert(isnumeric(rgb),...
	'SC:sRGB_to_CIELab:rgb:NotNumeric',...
	'1st input <rgb> array must be numeric.')
assert(isreal(rgb),...
	'SC:sRGB_to_CIELab:rgb:ComplexValue',...
	'1st input <rgb> cannot be complex.')
assert(isz(end)==3,...
	'SC:sRGB_to_CIELab:rgb:InvalidSize',...
	'1st input <rgb> last dimension must have size 3 (e.g. Nx3 or RxCx3).')
rgb = reshape(rgb,[],3);
assert(all(rgb(:)>=0&rgb(:)<=1),...
	'SC:sRGB_to_CIELab:rgb:OutOfRange',...
	'1st input <rgb> values must be within the range 0<=rgb<=1')
%
if ~isfloat(rgb)
	rgb = double(rgb);
end
%
%% RGB2Lab %%
%
M = [... Standard sRGB to XYZ matrix:
	0.4124,0.3576,0.1805;...
	0.2126,0.7152,0.0722;...
	0.0193,0.1192,0.9505];
% source: IEC 61966-2-1:1999
%
% M = [... High-precision sRGB to XYZ matrix:
% 	0.4124564,0.3575761,0.1804375;...
% 	0.2126729,0.7151522,0.0721750;...
% 	0.0193339,0.1191920,0.9503041];
% source: <http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html>
%
wpt = [0.95047,1,1.08883]; % D65
%
% Approximately equivalent to this function, requires Image Toolbox:
%Lab = applycform(rgb,makecform('srgb2lab','AdaptedWhitePoint',wpt))
%
% RGB2XYZ
if nargin>1 && isequal(isz,0:3)
	isz = size(test);
	XYZ = test;
else
	XYZ = sGammaInv(rgb) * M.';
end
%
% XYZ2Lab
epsilon = 216/24389;
kappa = 24389/27;
% source: <http://www.brucelindbloom.com/index.html?LContinuity.html>
xyzr = bsxfun(@rdivide,XYZ,wpt);
idx  = xyzr>epsilon;
fxyz = (kappa*xyzr+16)/116;
fxyz(idx) = nthroot(xyzr(idx),3);
Lab = reshape([max(0,min(100,...
	116*fxyz(:,2)-16)),...
	500*(fxyz(:,1)-fxyz(:,2)),...
	200*(fxyz(:,2)-fxyz(:,3))],isz);
% source: <http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Lab.html>
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%sRGB_to_CIELab
function out = sGammaInv(inp)
% Inverse gamma correction: Nx3 sRGB -> Nx3 linear RGB.
idx = inp > 0.04045;
out = inp / 12.92;
out(idx) = real(power((inp(idx) + 0.055) / 1.055, 2.4));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%sGammaInv