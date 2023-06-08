function [Lab,XYZ] = sRGB_to_OKLab(rgb,test)
% Convert a matrix of sRGB R G B values to OKLAB L a b values.
%
% (c) 2018-2023 Stephen Cobeldick
%
%%% Syntax:
% Lab = sRGB_to_OKLab(rgb)
%
% <https://bottosson.github.io/posts/oklab/>
%
%% Inputs and Outputs
%
%%% Input Argument:
% rgb = Numeric Array, size Nx3 or RxCx3, where the last dimension
%       encodes sRGB values [R,G,B] in the range 0<=RGB<=1.
%
%%% Output Argument:
% Lab = Numeric Array, same size as <rgb>, where the last dimension
%       encodes OKLAB values [L,a,b] in the range 0<=L<=1.
%
% See also SRGB_TO_CAM02UCS SRGB_TO_OSAUCS SRGB_TO_CIELAB
% CIELAB_TO_SRGB CIELAB_TO_DIN99 CIELAB_TO_DIN99O
% MAXDISTCOLOR MAXDISTCOLOR_VIEW MAXDISTCOLOR_DEMO

%% Input Wrangling %%
%
isz = size(rgb);
assert(isnumeric(rgb),...
	'SC:sRGB_to_OKLab:rgb:NotNumeric',...
	'1st input <rgb> array must be numeric.')
assert(isreal(rgb),...
	'SC:sRGB_to_OKLab:rgb:ComplexValue',...
	'1st input <rgb> cannot be complex.')
assert(isz(end)==3,...
	'SC:sRGB_to_OKLab:rgb:InvalidSize',...
	'1st input <rgb> last dimension must have size 3 (e.g. Nx3 or RxCx3).')
rgb = reshape(rgb,[],3);
assert(all(rgb(:)>=0&rgb(:)<=1),...
	'SC:sRGB_to_OKLab:rgb:OutOfRange',...
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
% RGB2XYZ
if nargin>1 && isequal(isz,0:3)
	isz = size(test);
	XYZ = test;
else
	XYZ = sGammaInv(rgb) * M.';
end
%
% XYZ2OKLab
M1 = [... XYZ to approximate cone responses:
	+0.8189330101, +0.3618667424, -0.1288597137;...
	+0.0329845436, +0.9293118715, +0.0361456387;...
	+0.0482003018, +0.2643662691, +0.6338517070];
M2 = [... nonlinear cone responses to Lab:
	+0.2104542553, +0.7936177850, -0.0040720468;...
	+1.9779984951, -2.4285922050, +0.4505937099;...
	+0.0259040371, +0.7827717662, -0.8086757660];
lms = XYZ * M1.';
lmsp = nthroot(lms,3);
Lab = lmsp * M2.';
Lab = reshape(Lab,isz);
% source: <https://bottosson.github.io/posts/oklab/>
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%sRGB_to_OKLab
function out = sGammaInv(inp)
% Inverse gamma correction: Nx3 sRGB -> Nx3 linear RGB.
idx = inp > 0.04045;
out = inp / 12.92;
out(idx) = real(power((inp(idx) + 0.055) / 1.055, 2.4));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%sGammaInv