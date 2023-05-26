function [rgb,ucs,status,RGB,UCS] = maxdistcolor(N,fun,opts,varargin)
% Generate an RGB colormap of maximally distinct colors, using a uniform colorspace.
%
% (c) 2017-2023 Stephen Cobeldick.
%
%%% Syntax:
% rgb = maxdistcolor(N,fun)
% rgb = maxdistcolor(N,fun,opts)
% rgb = maxdistcolor(N,fun,<name-value pairs>)
% [rgb,ucs,status] = maxdistcolor(N,fun,...)
%
% Repeatedly applies a greedy algorithm to search the RGB gamut to find
% a maximally distinct set of N colors. Note that requesting many colors
% from a large gamut can require hours/days/... of processing!
%
%%% Options include:
% * Limit the lightness range of the output colors.
% * Limit the chroma range of the output colors.
% * Colors to be excluded (e.g. background colors).
% * Colors to be included (e.g. corporate colors).
% * Specify the class (double/single) used for the RGB calculations.
% * Specify a different RGB bit depth (e.g. 8 bits per channel TrueColor).
% * Sort the output colormap (e.g. by hue, lightness, farthest colors, etc.).
%
% If intended for printing then the lightness and chroma ranges must be
% limited to suitable ranges (e.g. check the printing device's ICC profile).
%
%% Options %%
%
% The options may be supplied either
% 1) in a scalar structure, or
% 2) as a comma-separated list of name-value pairs.
%
% Field names and string values are case-insensitive. The following field
% names and values are permitted as options (**=default value):
%
% Field  | Permitted  |
% Name:  | Values:    | Description:
% =======|============|====================================================
% Lmin   | 0<=L<=1    | Lightness range limits to exclude light/dark colors.
% Lmax   |            | Scaled so 0==black and 1==white. Lmin=0**, Lmax=1**
% -------|------------|----------------------------------------------------
% Cmin   | 0<=C<=1    | Chroma range limits to exclude grays/saturated colors.
% Cmax   |            | Scaled so 1==max(gamut chroma). Cmin=0**, Cmax=1**
% -------|------------|----------------------------------------------------
% inc    | RGB matrix | Mx3 RGB matrix of colors to include. []**
% -------|------------|----------------------------------------------------
% exc    | RGB matrix | Mx3 RGB matrix of colors to exclude. [0,0,0;1,1,1]**
% -------|------------|----------------------------------------------------
% disp   | 'off'   ** | Does not print to the command window.
%        | 'time'     | Print the time required to complete the function.
%        | 'summary'  | Print the status after the completion of main steps.
%        | 'verbose'  | Print the status after every algorithm iteration.
% -------|------------|----------------------------------------------------
% sort   | 'none'  ** | The output matrices are not sorted.
%        | 'farthest' | The next color is the farthest from the current color.
%        | 'lightness'| Sorted by the lightness value L or J, i.e. ucs(:,1).
%        | 'a' or 'j' | Sorted by the a or j dimension value, i.e. ucs(:,2).
%        | 'b' or 'g' | Sorted by the b or g dimension value, i.e. ucs(:,3).
%        | 'chroma'   | Sorted by chroma (radius calculated from ucs(:,2:3)).
%        | 'hue'      | Sorted by hue (the angle calculated from ucs(:,2:3)).
%        | 'zip'      | Sorted by hue, then zip together alternating elements.
%        | 'maxmin'   | Maximize the minimum adjacent color difference. See Note1.
%        | 'minmax'   | Minimize the maximum adjacent color difference. See Note1.
%        | 'longest'  | The longest  path joining all color nodes.      See Note1.
%        | 'shortest' | The shortest path joining all color nodes.      See Note1.
% -------|------------|----------------------------------------------------
% path   | 'open'  ** | For <sort> options marked "See Note1" this selects
%        | 'closed'   | if the path forms a closed loop through all colors.
% -------|------------|----------------------------------------------------
% start  | 0<=A<=360  | Start angle for <sort> options 'hue' & 'zip'. 0**
% -------|------------|----------------------------------------------------
% class  | 'double'** | Specify the class used for the sRGB values matrix,
%        | 'single'   | the conversion function <fun> must accept this class.
% -------|------------|----------------------------------------------------
% bitR   | 1<=R<=53   | Color depth for sampling the   Red channel, 6** See Note2.
% bitG   | 1<=G<=53   | Color depth for sampling the Green channel, 7** See Note2.
% bitB   | 1<=B<=53   | Color depth for sampling the  Blue channel, 6** See Note2.
% -------|------------|----------------------------------------------------
%
% Note1: These algorithms use an exhaustive search which generates all row
%        permutations of the colormap, an error is thrown for N greater than 9.
% Note2: Using 8 bits per channel requires 64 bit MATLAB with atleast 8 GB RAM.
%        A smaller number of bits gives a smaller RGB gamut (faster), but the
%        greedy algorithm can fail to work for smaller RGB gamuts: user beware!
% Note3: Text names and values may be character vector or string scalar.
%
%% Examples %%
%
% >> N = 5;
% >> fun = @sRGB_to_OKLab;
% >> rgb = maxdistcolor(N,fun)
% rgb =
%          0         0    0.8095
%     1.0000    0.3858         0
%     0.8254         0    1.0000
%     0.4286         0    0.0159
%          0    0.8189         0
% >> axes('ColorOrder',rgb, 'NextPlot','replacechildren')
% >> X = linspace(0,pi*3,1000);
% >> Y = bsxfun(@(x,n)n*sin(x+2*n*pi/N), X(:), 1:N);
% >> plot(X,Y, 'linewidth',4)
%
% >> maxdistcolor(5,fun, 'exc',[0,0,0]) % Exclude black (e.g. background).
% ans =
%     1.0000    1.0000    1.0000
%          0         0    1.0000
%          0    0.6772         0
%     1.0000    0.1969    1.0000
%     0.5238         0    0.0635
%
% >> maxdistcolor(5,fun, 'inc',[1,0,1]) % Include magenta.
% ans =
%     1.0000         0    1.0000        % <- magenta!
%          0    0.8583         0
%     0.1111         0    1.0000
%          0    0.2677         0
%     0.8254    0.3858         0
%
% >> [rgb,Lab] = maxdistcolor(6,@sRGB_to_CIELab, 'Lmin',0.5, 'Lmax',0.7)
% rgb =
%     0.7619         0    1.0000
%     1.0000         0         0
%          0    0.7795         0
%          0    0.5591    1.0000
%     0.8254    0.6457    0.0794
%     0.8254    0.2835    0.5397
% Lab =
%    50.3665   89.7885  -77.4167
%    53.2329   80.1093   67.2201
%    69.9972  -71.4464   68.9566
%    58.7262    9.8285  -64.4614
%    69.8987    5.1700   70.3789
%    52.1378   59.8754   -6.6658
%
%% Input and Output Arguments %%
%
%%% Inputs:
%  N    = ScalarNumeric, the requested number of output colors.
%  fun  = FunctionHandle, a function to convert from RGB to a uniform colorspace.
%         The function must accept an Nx3 RGB matrix with values 0<=RGB<=1, and
%         return an Nx3 matrix in a uniform colorspace (UCS), where the columns
%         represent some version of [lightness,a,b], e.g. [L*,a*,b*] or [J',a',b'].
%  opts = ScalarStructure, with any field names and values as per 'Options' above.
%  OR
%  <name-value pairs> = a comma-separated list of names and corresponding values.
%
%%% Outputs:
%  rgb = NumericMatrix, size Nx3, the colors in RGB, where 0<=rgb<=1.
%  ucs = NumericMatrix, size Nx3, the colors in the uniform colorspace.
%  status = ScalarStructure of greedy algorithm status information.
%
% See also MAXDISTCOLOR_DEMO MAXDISTCOLOR_VIEW SRGB_TO_CAM02UCS SRGB_TO_OKLAB
% SRGB_TO_OSAUCS SRGB_TO_CIELAB CIELAB_TO_DIN99 CIELAB_TO_DIN99O
% TAB10 BREWERMAP LINES COLORMAP RGBPLOT AXES SET PLOT COLORNAMES

%% Input Wrangling %%
%
tbe = now();
%
assert(isnumeric(N)&&isscalar(N),...
	'SC:maxdistcolor:N:NotNumericScalar',...
	'First input <N> must be a numeric scalar.')
assert(isfinite(N)&&N>=0,...
	'SC:maxdistcolor:N:NotPositiveFinite',...
	'First input <N> must be a finite positive value. Input: %g',N)
assert(isreal(N),...
	'SC:maxdistcolor:N:ComplexValue',...
	'First input <N> cannot be a complex value. Input: %g%+gi',N,imag(N))
%
assert(isa(fun,'function_handle'),...
	'SC:maxdistcolor:fun:NotFunctionHandle',...
	'Second input <fun> must be a function handle.')
map = fun([0,0,0;1,1,1]);
assert(isfloat(map),...
	'SC:maxdistcolor:fun:OutputNotNumericArray',...
	'Second input <fun> must return a floating-point matrix.')
assert(isequal(size(map),2:3),...
	'SC:maxdistcolor:fun:OutputInvalidSize',...
	'Second input <fun> output matrix has an incorrect size.')
assert(isreal(map),...
	'SC:maxdistcolor:fun:OutputComplexValue',...
	'Second input <fun> output matrix values cannot be complex.')
assert(all(isfinite(map(:))),...
	'SC:maxdistcolor:fun:OutputNotFiniteValue',...
	'Second input <fun> output matrix values must be finite.')
%
% Default option values:
stpo = struct('start',0, 'Cmin',0, 'Cmax',1, 'Lmin',0, 'Lmax',1,...
	'class','double', 'sort','none', 'path','open', 'disp','off',...
	'exc',[0,0,0;1,1,1], 'inc',[], 'bitR',6, 'bitG',7, 'bitB',6);
%
% Check any user-supplied option fields and values:
if nargin==3
	assert(isstruct(opts)&&isscalar(opts),...
		'SC:maxdistcolor:opts:NotScalarStruct',...
		'When calling with three inputs, the third input <opts> must be a scalar structure.')
	opts = structfun(@mdc1s2c,opts,'UniformOutput',false);
	opts = mdcOptions(stpo,opts);
elseif nargin>3 % options as <name-value> pairs
	varargin = cellfun(@mdc1s2c,varargin,'UniformOutput',false);
	opts = struct(mdc1s2c(opts),varargin{:});
	assert(isscalar(opts),...
		'SC:maxdistcolor:opts:CellArrayValue',...
		'Invalid <name-value> pairs: cell array values are not permitted.')
	opts = mdcOptions(stpo,opts);
else
	opts = stpo;
end
stpo = opts;
%
stpo.ohm = pow2(cast([stpo.bitR,stpo.bitG,stpo.bitB],stpo.class))-1;
stpo.cyc = strcmpi('closed',stpo.path); % cyc: closed/open -> true/false.
stpo.mfn = mfilename(); % tsv: off/time/summary/verbose -> 0/1/2/3.
[~,stpo.tsv] = ismember(lower(stpo.disp),{'time','summary','verbose'});
%
mdcDisplay(stpo,1,'Starting...')
%
%% Generate RGB and UCS Arrays %%
%
assert(stpo.Lmax>stpo.Lmin,...
	'SC:maxdistcolor:opts:LmaxLessThanLmin',...
	'The value Lmax must be greater than the value Lmin.')
assert(stpo.Cmax>stpo.Cmin,...
	'SC:maxdistcolor:opts:CmaxLessThanCmin',...
	'The value Cmax must be greater than the value Cmin.')
%
[RGB,UCS] = mdcMakeGamut(stpo,fun);
%
% User supplied RGB colormaps:
exc = cast(reshape(stpo.exc.',3,[]).',stpo.class);
inc = cast(reshape(stpo.inc.',3,[]).',stpo.class);
cxe = fun(exc);
cni = fun(inc);
%
tmp = all(abs(bsxfun(@minus,exc,permute(inc,[3,2,1])))<1e-9,2);
assert(~any(tmp(:)),...
	'SC:maxdistcolor:opts:ExcAndIncOverlap',...
	'Options <exc> and <inc> must not contain the same RGB values')
%
cnt = 0;
nmg = 0+size(RGB,1); % number of color nodes in the RGB gamut.
nmt = N+size(exc,1); % number of color nodes to test (N + excluded colors).
nmf = N-size(inc,1); % number of color nodes to find (N - included colors).
%
dgt = 1+fix(log10(nmf));
win = [zeros(nmf,3,stpo.class);cxe;cni];
%
%% Greedy Algorithm %%
%
if nmf==0
	rgb = inc;
	ucs = cni;
	nin = nan(0,3);
elseif nmf>0
	assert(nmt<=nmg,'SC:maxdistcolor:opts:GamutTooSmall',...
		['The specified RGB gamut contains fewer color nodes than the\n',...
		'requested number of output colors. Ways to avoid this error:\n',...
		'* request fewer colors <N> (now: %u),\n',...
		'* decrease the number of <exc> colors (now: %u),\n',...
		'* increase the difference between <Cmax> and <Cmin>,\n',...
		'* increase the difference between <Lmax> and <Lmin>,\n',...
		'* increase the number of bits for any color channel.\n',...
		'The specified RGB gamut contains %u color nodes.'], N,nmt-N,nmg)
	%
	idz = zeros(nmf,1);
	chk = zeros(nmf,nmf);
	%
	mdcDisplay(stpo,2,'Starting the greedy algorithm...')
	%
	%dwn = nmf;
	vec = Inf;
	mxi = Inf;
	err = true;
	while err && cnt<=mxi
		row = 1+mod(cnt,nmf);
		cnt = 1+cnt;
		vec(:) = Inf;
		% Distance between all color nodes (except for the one being moved):
		for k = [1:row-1,row+1:nmt]
			%vec = min(vec,sum(bsxfun(@minus,ucs,win(k,:)).^2,2));
			vec = min(vec,... maybe uses less memory:
				(UCS(:,1)-win(k,1)).^2 + ...
				(UCS(:,2)-win(k,2)).^2 + ...
				(UCS(:,3)-win(k,3)).^2);
		end
		% Move that color node to the farthest point from the other nodes:
		[~,idr]    = max(vec);   % farthest point.
		win(row,:) = UCS(idr,:); % move node.
		idz(row,:) = idr; % Save the index.
		chk(row,:) = idz; % Save the index.
		% Check if any color nodes have changed index:
		tmp = any(diff(chk,1,1),1);
		err = any(tmp);
		%dwn = max(dwn-~err,dwn*err);
		%
		% Display:
		if stpo.tsv>=3
			fprintf('%s:  %*u %6u/%u %9.5f  %s\n', stpo.mfn,...
				dgt, row, cnt, mxi, mdcClosest(win), sprintf('%u',tmp))
		end
	end
	%
	rgb = [inc;RGB(idz,:)];
	ucs = [cni;UCS(idz,:)];
	nin =      UCS(idz,:);
	%
	mdcDisplay(stpo,3,'... optimized in %u iterations (metric=%g)...',cnt,mdcClosest(win))
	%
	mdcDisplay(stpo,2,'Finished the greedy algorithm in %u iterations.',cnt)
	%
else
	error('SC:maxdistcolor:opts:IncMoreColorsThanN',...
		'Not enough colors requested: option <inc> must have <N> or fewer rows.')
end
%
%% Sort %%
%
mdcDisplay(stpo,2,'Starting sorting the colormap...')
%
switch lower(stpo.sort)
	case 'none'
		ids = 1:N;
	case 'maxmin'
		ids = mdcBestPerm(ucs,N, stpo, @(v)-min(v));
	case 'minmax'
		ids = mdcBestPerm(ucs,N, stpo, @(v)+max(v));
	case 'longest'
		ids = mdcBestPerm(ucs,N, stpo, @(v)-sum(v));
	case 'shortest'
		ids = mdcBestPerm(ucs,N, stpo, @(v)+sum(v));
	case 'farthest'
		ids = mdcFarthest(ucs,N);
	case {'l','lightness'}
		[~,ids] = sortrows(ucs,1);
	case {'a','j'}
		[~,ids] = sortrows(ucs,2);
	case {'b','g'}
		[~,ids] = sortrows(ucs,3);
	case 'chroma'
		[~,ids] = sort(sum(ucs(:,2:3).^2,2));
	case 'hue'
		[~,ids] = sort(mdcAtan2D(ucs(:,3),ucs(:,2),stpo.start));
	case 'zip'
		[~,ids] = sort(mdcAtan2D(ucs(:,3),ucs(:,2),stpo.start));
		ids([1:2:N,2:2:N]) = ids;
	otherwise
		error('SC:maxdistcolor:sort:UnknownValue',...
			'This <sort> option is not supported: "%s".',typ)
end
%
rgb = rgb(ids,:);
ucs = ucs(ids,:);
%
mdcDisplay(stpo,2,'Finished sorting the colormap in "%s" order.',stpo.sort)
%
%% Status and Time %%
%
toa = (now()-tbe)*24*60*60;
%
if nargout>2
	status = struct();
	status.seconds = toa;
	status.options = opts;
	status.gamutSize = nmg;
	status.iterations = cnt;
	status.minDistOutput = mdcClosest(ucs);
	status.minDistAndExc = mdcClosest(win);
	status.minDistNotInc = mdcClosest(nin);
	[status.colorspace,status.axesLabels] = mdcColorspace(fun);
end
%
if ~stpo.tsv
	return
end
%
spl = nan(1,4);
dpf = 100;
toa = ceil(toa*dpf)/dpf;
spl(4) = rem(toa,60); % seconds
toa    = fix(toa/60);
spl(3) = rem(toa,60); % minutes
toa    = fix(toa/60);
spl(2) = rem(toa,24); % hours
toa    = fix(toa/24);
spl(1) =     toa    ; % days
idt = spl~=0 | [false(1,3),~any(spl)];
idp = spl~=1 & idt;
fmt = {' %d day',' %d hour',' %d minute',' %g second';'s','s','s','s'};
str = sprintf([fmt{[idt;idp]}],spl(idt));
%
fprintf('%s: Finished everything in%s.\n',stpo.mfn,str);
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%maxdistcolor
function stpo = mdcOptions(stpo,opts)
% Options check: only supported fieldnames with suitable option values.
%
dfc = fieldnames(stpo);
ofc = fieldnames(opts);
idc = strcmpi(ofc,'class');
ofc = [ofc(idc);ofc(~idc)];
%
for k = 1:numel(ofc)
	ofn = ofc{k};
	idf = strcmpi(ofn,ofc);
	if nnz(idf)>1
		error('SC:maxdistcolor:options:DuplicateOptionNames',...
			'Duplicate field names:%s\b.',sprintf(' <%s>,',ofc{idf}))
	end
	arg = opts.(ofn);
	dfn = lower(ofn);
	switch dfn
		case {'exc','inc'}
			mdcColormap()
		case {'bitr','bitg','bitb'}
			mdcScalar(false, 1,53)
		case {'cmin','cmax','lmax','lmin'}
			mdcScalar(true, 0,1)
		case 'start'
			mdcScalar(true, 0,360)
		case 'class'
			mdcString('single','double')
		case 'disp'
			mdcString('none','off','time','summary','verbose')
		case 'path'
			mdcString('open','closed')
		case 'sort'
			mdcString('none',...
				'maxmin','minmax','longest','shortest',... <- all permutations!
				'farthest','lightness','l','a','b','j','g','chroma','hue','zip');
		otherwise
			dfs = sort(dfc);
			error('SC:maxdistcolor:options:UnknownOptionName',...
				'Unknown option:%s\b.\nKey/Option names must be:%s\b.',...
				ofn,sprintf(' <%s>,',dfs{:}))
	end
	stpo.(dfn) = arg;
end
%
%% Nested Functions %%
%
	function mdcString(varargin) % text.
		if ~mdcIsCRV(arg)||~any(strcmpi(arg,varargin))
			tmp = sprintf(' <%s>,',varargin{:});
			error('SC:maxdistcolor:str:UnknownOptionValue',...
				'The <%s> value must be one of:%s\b.',dfn,tmp);
		end
		arg = lower(arg);
	end
	function mdcScalar(isf,minV,maxV) % numeric scalar
		dfn = dfc{strcmpi(ofn,dfc)};
		assert(isnumeric(arg),...
			'SC:maxdistcolor:val:NotNumeric',...
			'The <%s> input must be numeric. Class: %s',dfn,class(arg))
		assert(isscalar(arg),...
			'SC:maxdistcolor:val:NotScalar',...
			'The <%s> input must be scalar. Numel: %u',dfn,numel(arg))
		assert(imag(arg)==0,...
			'SC:maxdistcolor:val:ComplexValue',...
			'The <%s> value cannot be complex. Input: %g%+gi',dfn,arg,imag(arg))
		assert(isf||(fix(arg)==arg),...
			'SC:maxdistcolor:val:NotInteger',...
			'The <%s> value must be integer. Input: %g',dfn,arg)
		assert(arg>=minV,...
			'SC:maxdistcolor:val:AboveRange',...
			'The <%s> value must be >=%g. Input: %g',dfn,minV,arg)
		assert(arg<=maxV,...
			'SC:maxdistcolor:val:BelowRange',...
			'The <%s> value must be <=%g. Input: %g',dfn,maxV,arg)
		arg = double(arg);
	end
	function mdcColormap() % Nx3 colormap.
		assert(isnumeric(arg),...
			'SC:maxdistcolor:map:NotNumeric',...
			'The <%s> input must be a numeric matrix.',dfn)
		assert(ndims(arg)==2,...
			'SC:maxdistcolor:map:NotMatrix',...
			'The <%s> input must be a numeric matrix ',dfn) %#ok<ISMAT>
		if isequal(arg,[])
			return
		end
		assert(size(arg,2)==3,...
			'SC:maxdistcolor:map:InvalidSize',...
			'The <%s> input must have size Nx3, or be [].',dfn)
		assert(all(isfinite(arg(:))),...
			'SC:maxdistcolor:map:NotFiniteValue',...
			'The <%s> input must be finite RGB values.',dfn)
		assert(isreal(arg),...
			'SC:maxdistcolor:map:ComplexValue',...
			'The <%s> input cannot be complex.',dfn)
		assert(all(arg(:)>=0&arg(:)<=1),...
			'SC:maxdistcolor:map:OutOfRange',...
			'The <%s> input must contain values 0<=RGB<=1',dfn)
		arg = cast(arg,stpo.class);
	end
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%mdcOptions
function mdcDisplay(stpo,val,fmt,varargin)
% Display text in the command window.
if stpo.tsv>=val
	fprintf('%s: %s\n',stpo.mfn,sprintf(fmt,varargin{:}))
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%mdcDisplay
function [RGB,UCS] = mdcMakeGamut(stpo,fun)
% Generate RGB cube and convert to UCS, then select by lightness & chroma.
%
mdcDisplay(stpo,2,'Starting RGB cube creation...')
% Generate all RGB colors in the RGB cube:
z = zeros(1,1,stpo.class);
[R,G,B] = ndgrid(...
	z:stpo.ohm(1),...
	z:stpo.ohm(2),...
	z:stpo.ohm(3));
RGB = bsxfun(@rdivide,[R(:),G(:),B(:)],stpo.ohm);
clear R G B
mdcDisplay(stpo,2,'Finished RGB cube creation of %u colors.',size(RGB,1))
% Convert to uniform colorspace (e.g. L*a*b* or J'a'b'):
mdcDisplay(stpo,2,'Starting conversion from RGB to UCS (calling external function)...')
UCS = fun(RGB);
mdcDisplay(stpo,2,'Finished conversion from RGB to UCS.')
% Identify lightness and chroma values within the requested ranges:
mdcDisplay(stpo,2,'Starting gamut selection (RGB subsample using UCS lightness and chroma)...')
tmp = fun([0,0,0;1,1,1]);
lim = interp1([0;1],tmp(:,1),[stpo.Lmin;stpo.Lmax]);
chr = sqrt(sum(UCS(:,2:3).^2,2));
cim = interp1([0;1],[0,max(chr)],[stpo.Cmin;stpo.Cmax]);
idg = UCS(:,1)>=lim(1) & UCS(:,1)<=lim(2) & chr>=cim(1) & chr<=cim(2);
% Select gamut colors with the requested lightness and chroma values:
RGB = RGB(idg,:);
UCS = UCS(idg,:);
mdcDisplay(stpo,2,'Finished gamut selection of %u colors.',nnz(idg))
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%mdcMakeGamut
function ang = mdcAtan2D(Y,X,start)
% ATAN2 with an output in degrees. Note: ATAN2D only introduced R2012b.
ang = mod(360*atan2(Y,X)/(2*pi)-start,360);
ang(Y==0 & X==0) = 0;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%mdcAtan2D
function idp = mdcBestPerm(ucs,N,stpo,cost)
% Exhaustive search for the permutation that minimizes the cost function.
%
assert(N<10,'SC:maxdistcolor:sort:TooManyPermutations',...
	'<sort> option "%s" requires N<10, as it generates all permutations.',stpo.sort)
%
big = prod(1:N);
cyc = stpo.cyc;
tsv = stpo.tsv>=3;
low = cost(sum(diff([ucs;ucs(cyc,:)],1,1).^2,2));
if tsv
	fprintf('%s: %9d/%-9d  %g\n',stpo.mfn,1,big,low)
end
% Generate permutations using Heap's algorithm:
idx = 1:N;
idp = idx;
idc = ones(1,N);
idi = 1;
cnt = 1;
trk = 1;
while idi<=N
	if idc(idi)<idi
		cnt = cnt+1;
		% Swap indices:
		vec = [idc(idi),1];
		tmp = vec(1+mod(idi,2));
		idx([tmp,idi]) = idx([idi,tmp]);
		% Calculate the cost:
		new = cost(sum(diff(ucs([idx,idx(cyc)],:),1,1).^2,2));
		if new<low
			low = new;
			idp = idx;
			trk = cnt;
		end
		if tsv
			fprintf('%s: %9d/%-9d  %g\n',stpo.mfn,cnt,big,low)
		end
		% Prepare next iteration:
		idc(idi) = 1+idc(idi);
		idi      = 1;
	else
		idc(idi) = 1;
		idi      = 1+idi;
	end
end
%
mdcDisplay(stpo,3,'... checked %u permutations (best=%u, metric=%g)...',big,trk,low)
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%mdcBestPerm
function idp = mdcFarthest(ucs,N)
% Permutation where the next color is the farthest of the remaining colors.
%
dst = sum(bsxfun(@minus,permute(ucs,[1,3,2]),permute(ucs,[3,1,2])).^2,3);
[~,idx] = max(sum(dst));
idp = [idx,2:N];
for k = 2:N
	vec = dst(:,idx);
	dst(idx,:) = -Inf;
	dst(:,idx) = -Inf;
	[~,idx] = max(vec);
	idp(k) = idx;
end
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%mdcFarthest
function dst = mdcClosest(ucs)
% Distance between the closest pair of colors.
%
dst = Inf;
for k = 2:size(ucs,1)
	idr = 1:k-1;
	dst = min(dst,min(...
		(ucs(k,1)-ucs(idr,1)).^2 + ...
		(ucs(k,2)-ucs(idr,2)).^2 + ...
		(ucs(k,3)-ucs(idr,3)).^2));
end
dst = sqrt(dst);
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%mdcClosest
function [csn,lbl] = mdcColorspace(fnh)
% Colorspace name and axes labels:
fns = func2str(fnh);
tkn = regexp(fns,'\w+_to_(\w+)[^''"]*\W?([SL]CD|UCS)?','once','tokens');
css = struct('CAM02UCS',{'J''','a''','b'''}, 'CIELAB',{'L*','a*','b*'},...
	'DIN99O',{'L_{99o}','a_{99o}','b_{99o}'}, 'OKLAB',{'L','a','b'},...
	'DIN99',{'L_{99}','a_{99}','b_{99}'}, 'OSAUCS',{'L','j','g'});
try
	csn = [tkn{2},'UCS'];
	csn = strrep(tkn{1},'UCS',csn(1:3));
	lbl = {css.(upper(tkn{1}))};
catch
	csn = '???';
	lbl = {'L?','a?','b?'};
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%mdcColorspace
function out = mdcIsCRV(txt)
% TXT is character row vector.
szv = size(txt);
out = ischar(txt) && numel(szv)==2 && szv(1)==1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%mdcIsCRV
function arr = mdc1s2c(arr)
% If scalar string then extract the character vector, otherwise data is unchanged.
if isa(arr,'string') && isscalar(arr)
	arr = arr{1};
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%mdc1s2c