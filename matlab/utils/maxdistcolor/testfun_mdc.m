function chk = testfun_mdc(fnh,dnm,tol)
% Support function for comparing function output against expected output.
%
% (c) 2017-2023 Stephen Cobeldick
%
% See also TEST_CIELAB TEST_DIN99X TEST_OKLAB TEST_OSAUCS

chk = @nestfun;
itr = 0;
cnt = 0;
fnt = func2str(fnh);
fmn = ' %+#.17g(%s)';
if feature('hotlinks')
	fmt = '<a href="matlab:opentoline(''%1$s'',%2$d)">%3$s|%2$d:</a>';
else
	fmt = '%3$s|%2$d:';
end
%
	function nestfun(varargin)
		% (in1,in2,..,fnh,out1,out2,..)
		%
		dbs = dbstack();
		%
		if ~nargin % post-processing
			fprintf(fmt, dbs(2).file, dbs(2).line, fnt);
			fprintf(' %d of %d testcases failed.\n',cnt,itr);
			return
		end
		%
		idx = find(cellfun(@(f)isequal(f,fnh),varargin));
		assert(nnz(idx)==1,'SC:test_fun:MissFun','Missing/duplicated function handle.')
		xpc = varargin(idx+1:end);
		opc = cell(size(xpc));
		%
		[opc{:}] = fnh(varargin{1:idx-1});
		%
		ido = ~cellfun(@(a)isequal(a,@i),xpc);
		assert(nnz(ido)==1,'SC:test_fun:OneOutput','Select exactly one output.')
		%
		opa = opc{ido};
		xpa = xpc{ido};
		%
		boo = false;
		%
		if ~isequal(class(opa),class(xpa))
			boo = true;
			xpt = sprintf(' class %s',class(xpa));
			opt = sprintf(' class %s',class(opa));
		elseif ~isequal(size(opa),size(xpa))
			boo = true;
			xpt = sprintf(',%d',size(xpa));
			opt = sprintf(',%d',size(opa));
			xpt = sprintf(' size [%s]',xpt(2:end));
			opt = sprintf(' size [%s]',opt(2:end));
		elseif any(abs(opa(:)-xpa(:))>tol)
			boo = true;
			xpu = [num2cell(xpa(:).');dnm];
			opu = [num2cell(opa(:).');dnm];
			xpt = sprintf(fmn,xpu{:});
			opt = sprintf(fmn,opu{:});
		end
		%
		if boo
			dmn = min(numel(opt),numel(xpt));
			dmx = max(numel(opt),numel(xpt));
			dtx = repmat('^',1,dmx);
			dtx(opt(1:dmn)==xpt(1:dmn)) = ' ';
			%
			fprintf(fmt, dbs(2).file, dbs(2).line, fnt);
			fprintf(' (output #%d)\n',find(ido));
			fprintf('actual: %s\n', opt)
			fprintf('expect: %s\n', xpt)
			fprintf('diff:   ')
			fprintf(2,'%s\n',dtx); % red!
		end
		%
		cnt = cnt+boo;
		itr = itr+1;
	end
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%testfun_mdc