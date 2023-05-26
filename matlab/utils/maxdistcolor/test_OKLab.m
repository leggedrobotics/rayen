function test_OKLab()
% Test OKLab conversions against reference values.
%
% (c) 2017-2023 Stephen Cobeldick
%
% See also TESTFUN_MDC SRGB_TO_OKLAB OKLAB_TO_SRGB

fprintf('Running %s...\n',mfilename)
%
ist = nan(0:3);
fnh = @sRGB_to_OKLab;
%
% Reference: <https://bottosson.github.io/posts/oklab/>
chk = testfun_mdc(fnh,{'L','a','b'},5e-4);
chk(ist,[1,0,0], fnh, [+0.450,+1.236,-0.019]) % XYZ->OKLab
chk(ist,[0,1,0], fnh, [+0.922,-0.671,+0.263]) % XYZ->OKLab
chk(ist,[0,0,1], fnh, [+0.153,-1.415,-0.449]) % XYZ->OKLab
chk()
%
% Reference: by hand
chk = testfun_mdc(fnh,{'L','a','b'},5e-15);
chk([1,0,0], fnh, [0.627925900661856, +0.224887603832926, +0.125804933240474]) % sRGB->OKLab
chk([0,1,0], fnh, [0.866451874607104, -0.233921435420743, +0.179421768053411]) % sRGB->OKLab
chk([0,0,1], fnh, [0.452032954406456, -0.032351637521107, -0.311620544208813]) % sRGB->OKLab
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%test_OSAUCS