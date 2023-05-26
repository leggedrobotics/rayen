function test_DIN99x()
% Test DIN99/DIN99o conversions against reference values.
%
% (c) 2017-2023 Stephen Cobeldick
%
% See also TESTFUN_MDC CIELAB_TO_DIN99 CIELAB_TO_DIN99O DIN99_TO_SRGB

fprintf('Running %s...\n',mfilename)
%
fnh = @CIELab_to_DIN99;
%
% Reference: <https://colour.readthedocs.io/en/develop/index.html>
chk = testfun_mdc(fnh,{'L99','a99','b99'},1e-3);
chk([41.52787529,52.63858304,26.92317922], fnh, [53.22821988,28.41634656,3.89839552]) % Lab->DIN99
chk()
%
% Reference: <https://de.wikipedia.org/wiki/DIN99-Farbraum>
chk = testfun_mdc(fnh,{'L99','a99','b99'},5e-3);
chk([ 50,+10,+10], fnh, [61.43, +9.70, +3.76]) % Lab->DIN99
chk([ 50,+50,+50], fnh, [61.43,+28.64,+11.11]) % Lab->DIN99
chk([ 50,-10,+10], fnh, [61.43, -5.57, +7.03]) % Lab->DIN99
chk([ 50,-50,+50], fnh, [61.43,-17.22,+21.75]) % Lab->DIN99
chk([ 50,-10,-10], fnh, [61.43, -9.70, -3.76]) % Lab->DIN99
chk([ 50,-50,-50], fnh, [61.43,-28.64,-11.11]) % Lab->DIN99
chk([ 50,+10,-10], fnh, [61.43, +5.57, -7.03]) % Lab->DIN99
chk([ 50,+50,-50], fnh, [61.43,+17.22,-21.75]) % Lab->DIN99
chk([  0,  0,  0], fnh, [    0,     0,     0]) % Lab->DIN99
chk([100,  0,  0], fnh, [  100,     0,     0]) % Lab->DIN99
chk()
%
fnh = @CIELab_to_DIN99o;
%
% Reference: <https://de.wikipedia.org/wiki/DIN99-Farbraum>
chk = testfun_mdc(fnh,{'L99o','a99o','b99o'},5e-4);
chk([ 50,+10,+10], fnh, [54.098,+12.215,+10.979]) % Lab->DIN99o
chk([ 50,+50,+50], fnh, [54.098,+31.237,+28.076]) % Lab->DIN99o
chk([ 50,-10,+10], fnh, [54.098,-11.067, +9.780]) % Lab->DIN99o
chk([ 50,-50,+50], fnh, [54.098,-29.384,+25.968]) % Lab->DIN99o
chk([ 50,-10,-10], fnh, [54.098,-12.215,-10.979]) % Lab->DIN99o
chk([ 50,-50,-50], fnh, [54.098,-31.237,-28.076]) % Lab->DIN99o
chk([ 50,+10,-10], fnh, [54.098,+11.067, -9.780]) % Lab->DIN99o
chk([ 50,+50,-50], fnh, [54.098,+29.384,-25.968]) % Lab->DIN99o
chk([  0,  0,  0], fnh, [     0,      0,      0]) % Lab->DIN99o
chk([100,  0,  0], fnh, [   100,      0,      0]) % Lab->DIN99o
chk()
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%test_DIN99x