function [Ljg,XYZ] = sRGB_to_OSAUCS(rgb,isd,isc,test)
% Convert a matrix of sRGB R G B values to OSA-UCS L j g values.
%
% (c) 2020-2023 Stephen Cobeldick
%
%%% Syntax:
% Ljg = sRGB_to_OSAUCS(rgb)
% Ljg = sRGB_to_OSAUCS(rgb,isd)
% Ljg = sRGB_to_OSAUCS(rgb,isd,isc)
%
% If the output is being used for calculating the Euclidean color distance
% (i.e. deltaE) use isd=true, so that L is NOT divided by sqrt(2).
%
% The reference formula divides by zero when Y0^(1/3)==2/3 (dark colors),
% this unfortunate numeric discontinuity can be avoided with isc=true.
%
% <https://en.wikipedia.org/wiki/SRGB>
% <https://en.wikipedia.org/wiki/OSA-UCS>
%
%% Inputs and Outputs
%
%%% Input Argument (*==default):
% rgb = NumericArray, size Nx3 or RxCx3, where the last dimension
%       encodes sRGB values [R,G,B] in the range 0<=RGB<=1.
% isd = LogicalScalar, true/false* = Euclidean distance/reference output values.
% isc = LogicalScalar, true/false* = modified continuous/reference output values.
%
%%% Output Argument:
% Ljg = Numeric Array, same size as <rgb>, where the last dimension
%       encodes OSA-UCS values [L,j,g].
%
% See also SRGB_TO_CAM02UCS SRGB_TO_OKLAB SRGB_TO_CIELAB
% CIELAB_TO_SRGB CIELAB_TO_DIN99 CIELAB_TO_DIN99O
% MAXDISTCOLOR MAXDISTCOLOR_VIEW MAXDISTCOLOR_DEMO

%% Input Wrangling %%
%
isz = size(rgb);
assert(isnumeric(rgb),...
	'SC:sRGB_to_OSAUCS:rgb:NotNumeric',...
	'1st input <rgb> array must be numeric.')
assert(isreal(rgb),...
	'SC:sRGB_to_OSAUCS:rgb:ComplexValue',...
	'1st input <rgb> cannot be complex.')
assert(isz(end)==3,...
	'SC:sRGB_to_OSAUCS:rgb:InvalidSize',...
	'1st input <rgb> last dimension must have size 3 (e.g. Nx3 or RxCx3).')
rgb = reshape(rgb,[],3);
assert(all(rgb(:)>=0&rgb(:)<=1),...
	'SC:sRGB_to_OSAUCS:rgb:OutOfRange',...
	'1st input <rgb> values must be within the range 0<=rgb<=1')
%
if ~isfloat(rgb)
	rgb = double(rgb);
end
%
assert(nargin<2||ismember(isd,0:1),...
	'SC:sRGB_to_OSAUCS:isd:NotScalarLogical',...
	'Second input <isd> must be true/false.')
ddd = 2-(nargin>1&&isd);
%
assert(nargin<3||ismember(isc,0:1),...
	'SC:sRGB_to_OSAUCS:isc:NotScalarLogical',...
	'Third input <isc> must be true/false.')
ccc = 30*(nargin>2&&isc);
%
%% RGB2Ljg %%
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
if nargin>3 && isequal(isz,0:3)
	isz = size(test);
	XYZ = test;
else
	XYZ = 100 * sGammaInv(rgb) * M.';
end
%
% XYZ2Ljg
xyz = bsxfun(@rdivide,XYZ,sum(XYZ,2));
xyz(isnan(xyz)) = 0;
%
K = 1.8103 + (xyz(:,1:2).^2)*[4.4934;4.3034] - ...
	prod(xyz(:,1:2),2)*4.276 - xyz(:,1:2)*[1.3744;2.5643];
Y0 = K.*XYZ(:,2);
Lp = 5.9*(nthroot(Y0,3)-2/3 + 0.042*nthroot(Y0-30,3));
L = (Lp-14.3993)./sqrt(ddd);
%C = 1 + (0.042*nthroot(Y0-30,3))./(nthroot(Y0,3)-2/3); % !!!!! divide by zero !!!!!
C = 1 + (0.042*nthroot(Y0-30,3))./(nthroot(max(ccc,Y0),3)-2/3);
tmp = nthroot(XYZ*[0.799,0.4194,-0.1648;-0.4493,1.3265,0.0927;-0.1149,0.3394,0.717].',3);
a = tmp*[-13.7;17.7;-4];
b = tmp*[1.7;8;-9.7];
Ljg = reshape([L,C.*b,C.*a],isz);
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%sRGB_to_OSAUCS
function out = sGammaInv(inp)
% Inverse gamma correction: Nx3 sRGB -> Nx3 linear RGB.
idx = inp > 0.04045;
out = inp / 12.92;
out(idx) = real(power((inp(idx) + 0.055) / 1.055, 2.4));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%sGammaInv