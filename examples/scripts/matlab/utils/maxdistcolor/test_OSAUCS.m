function test_OSAUCS()
% Test OSA-UCS conversions against reference values.
%
% (c) 2017-2023 Stephen Cobeldick
%
% See also TESTFUN_MDC SRGB_TO_OSAUCS OSAUCS_to_SRGB

fprintf('Running %s...\n',mfilename)
%
ist = nan(0:3);
fnh = @sRGB_to_OSAUCS;
%
% Reference: <https://colour.readthedocs.io/en/develop/index.html>
chk = testfun_mdc(fnh,{'L','j','g'},5e-4);
chk(ist,false,false, 100*[0.20654008,0.12197225,0.05136952], fnh, [-3.0049979,+2.99713697,-9.66784231]) % XYZ->OSAUCS
chk()
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%test_OSAUCS