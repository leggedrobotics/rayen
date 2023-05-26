function test_CIELab()
% Test CIELab conversions against sample values.
%
% (c) 2017-2023 Stephen Cobeldick
%
% See also TESTFUN_MDC SRGB_TO_CIELAB CIELAB_TO_SRGB

fprintf('Running %s...\n',mfilename)
%
ist = nan(0:3);
fnh = @sRGB_to_CIELab;
%
% Reference: <https://davengrace.com/dave/cspace/>
chk = testfun_mdc(fnh,{'X','Y','Z'},5e-16);
chk([0,0,0], fnh, @i, [0,0,0]) % sRGB->XYZ
chk([1,1,1], fnh, @i, [0.9505,1,1.089]) % sRGB->XYZ
chk([0.1,0.2,0.3], fnh, @i, [0.0291913093288767, 0.0310952343831350, 0.0737531562712688]) % sRGB->XYZ
chk([0.4,0.5,0.6], fnh, @i, [0.1888337010968200, 0.2043291062158660, 0.3308567751593710]) % sRGB->XYZ
chk([0.7,0.8,0.9], fnh, @i, [0.5428069959016100, 0.5839508165288460, 0.8290577762222190]) % sRGB->XYZ
chk([0.9,0.8,0.7], fnh, @i, [0.6215193929671810, 0.6316059288531500, 0.5129862620029020]) % sRGB->XYZ
chk([0.6,0.5,0.4], fnh, @i, [0.2318925351756980, 0.2303983615184730, 0.1579529964002000]) % sRGB->XYZ
chk([0.3,0.2,0.1], fnh, @i, [0.0438511299462668, 0.0399707790777515, 0.0148862957326157]) % sRGB->XYZ
chk([0.04,0.02,0.01], fnh, @i, [0.00197004643962848, 0.00182120743034056, 0.000979953560371517]) % sRGB->XYZ
chk([0.03,0.02,0.01], fnh, @i, [0.00165085139318885, 0.00165665634674923, 0.000965015479876161]) % sRGB->XYZ
chk([0.02,0.02,0.01], fnh, @i, [0.00133165634674923, 0.00149210526315789, 0.000950077399380805]) % sRGB->XYZ
chk([0.01,0.02,0.01], fnh, @i, [0.00101246130030960, 0.00132755417956656, 0.000935139318885449]) % sRGB->XYZ
chk()
%
% Reference: <https://davengrace.com/dave/cspace/> !!! not accurate sRGB->CIELab !!!
chk = testfun_mdc(fnh,{'L*','a*','b*'},1e-2); % sRGB->CIELab
chk([0.2,0.4,0.6], fnh, [42.0099857768665, -0.147373964354935, -32.8445986139017]) % sRGB->CIELab
chk([0.1,0.2,0.3], fnh, [20.4772929305063, -0.649398365518222, -18.63125696672030]) % sRGB->CIELab
chk([0.9,0.8,0.7], fnh, [83.5268033115757, +4.986300524367880, +15.98132781035020]) % sRGB->CIELab
chk([0.04,0.02,0.01], fnh, [1.64501872414861, +0.978923677309326,  +1.43484259764831]) % sRGB->CIELab
chk()
%
% Reference: <http://www.brucelindbloom.com/index.html?ColorCalculator.html>
chk = testfun_mdc(fnh,{'L*','a*','b*'},5e-5);
chk(ist, [0,0,0], fnh, [0,0,0]) % XYZ->CIELab
chk(ist, [0.1,0.2,0.3], fnh, [51.8372, -56.3591, -13.1812]) % XYZ->CIELab
chk(ist, [0.4,0.5,0.6], fnh, [76.0693, -22.1559,  -5.2284]) % XYZ->CIELab
chk(ist, [0.7,0.8,0.9], fnh, [91.6849, -12.6255,  -2.0335]) % XYZ->CIELab
chk(ist, [0.9,0.9,0.9], fnh, [95.9968,  +8.2439,  +5.4008]) % XYZ->CIELab
chk(ist, [0.9,0.8,0.7], fnh, [91.6849, +26.8297, +13.0496]) % XYZ->CIELab
chk(ist, [0.6,0.5,0.4], fnh, [76.0693, +32.0677, +15.5004]) % XYZ->CIELab
chk(ist, [0.3,0.2,0.1], fnh, [51.8372, +48.0307, +26.7254]) % XYZ->CIELab
chk(ist, [0.006,0.005,0.004], fnh, [4.5165, +5.1109, +2.0656]) % XYZ->CIELab
chk(ist, [0.003,0.002,0.001], fnh, [1.8066, +4.5022, +1.6845]) % XYZ->CIELab
chk(ist, [0.001,0.002,0.003], fnh, [1.8066, -3.6906, -1.1762]) % XYZ->CIELab
chk(ist, [0.004,0.005,0.006], fnh, [4.5165, -3.0819, -0.7951]) % XYZ->CIELab
chk()
%
% Reference: colorspacious.cspace_convert()
chk = testfun_mdc(fnh,{'L*','a*','b*'},5e-13);
chk([0,0,0], fnh, [0,0,0]) % sRGB->CIELab
% Note that COLORSPACIOUS does not use the sRGB->XYZ matrix defined in
% IEC 61966, but instead uses the inverse of the XYZ->sRGB matrix, which
% differs slightly after the fourth decimal digits. If we modify
% COLORSPACIOUS to use the standard matrix, these are the results:
chk([0.1,0.2,0.3], fnh, [20.47729293050626, -0.6477509840630280, -18.635499506210450]) % sRGB->CIELab
chk([0.4,0.5,0.6], fnh, [52.32317935858785, -2.7423177310761380, -16.660446906196704]) % sRGB->CIELab
chk([0.7,0.8,0.9], fnh, [80.95794732626959, -3.0900960941968036, -15.460893809249443]) % sRGB->CIELab
chk([0.9,0.8,0.7], fnh, [83.52680331157573, +4.9908664399307410, +15.973229378445343]) % sRGB->CIELab
chk([0.6,0.5,0.4], fnh, [55.11334675196463, +5.9069971684236890, +17.521347800346620]) % sRGB->CIELab
chk([0.3,0.2,0.1], fnh, [23.66177930857765, +8.3731659005595310, +20.559154151582720]) % sRGB->CIELab
chk([0.04,0.02,0.01], fnh, [1.6450899266139203, +0.97922076074241640, +1.4346858921472416]) % sRGB->CIELab
chk([0.03,0.02,0.01], fnh, [1.4964515422543272, +0.31234842724181533, +1.1997794911020432]) % sRGB->CIELab
chk([0.02,0.02,0.01], fnh, [1.3478131578947340, -0.35452390625879960, +0.9648730900568503]) % sRGB->CIELab
chk([0.01,0.02,0.01], fnh, [1.1991747735351446, -1.02139623975940050, +0.7299666890116518]) % sRGB->CIELab
chk()
% Unmodified COLORSPACIOUS results (quite different!):
chk = testfun_mdc(fnh,{'L*','a*','b*'},5e-3);
chk([0.1,0.2,0.3], fnh, [20.476868826608616, -0.6479456121413207, -18.635946031759400]) % sRGB->CIELab
chk([0.4,0.5,0.6], fnh, [52.322284675856210, -2.7417959374207435, -16.661347434860897]) % sRGB->CIELab
chk([0.7,0.8,0.9], fnh, [80.956635208316530, -3.0889674319084515, -15.462182842749028]) % sRGB->CIELab
chk([0.9,0.8,0.7], fnh, [83.525303672498030, +4.9934265610782070, +15.971981007339942]) % sRGB->CIELab
chk([0.6,0.5,0.4], fnh, [55.112248327000200, +5.9090692144755375, +17.520501311385940]) % sRGB->CIELab
chk([0.3,0.2,0.1], fnh, [23.661122238740510, +8.3747095232966360, +20.558842534531685]) % sRGB->CIELab
chk()
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%test_CIELab