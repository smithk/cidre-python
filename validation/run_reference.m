function run_reference(cidre_matlab_dir, infile, outfile)
% Run the ORIGINAL MATLAB CIDRE code on a stack and save its intermediate and
% final results for comparison with the Python port (see compare.py).
%
%   run_reference('../../cidre/trunk/matlab', 'data/cells.mat', 'data/cells_ref.mat')
%
% where ../../cidre is a clone of https://github.com/smithk/cidre
%
% Works in MATLAB as is. In GNU Octave (with the image package) the folder
% octave_shims/ supplies MATLAB-exact imresize and fspecial('log'), and minFunc
% is run without its MATLAB-only mex files.

root = cidre_matlab_dir;
here = fileparts(mfilename('fullpath'));
is_octave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

addpath([root '/3rdparty/minFunc2012/autoDif']);
addpath([root '/3rdparty/minFunc2012/minFunc']);
addpath([root '/io']); addpath([root '/gui']); addpath([root '/main']);

if is_octave
    pkg load image
    % minFunc's mex binaries cannot load in Octave: use a copy of
    % cdr_cidreModel that sets useMex = 0 (identical maths, slower).
    patched = tempname(); mkdir(patched);
    src = fileread([root '/main/cdr_cidreModel.m']);
    src = strrep(src, 'mf_opt.Corr         = 100;', 'mf_opt.Corr = 100; mf_opt.useMex = 0;');
    fid = fopen([patched '/cdr_cidreModel.m'], 'w'); fwrite(fid, src); fclose(fid);
    addpath(patched);
    addpath([here '/octave_shims']);   % must come first on the path
else
    addpath([root '/3rdparty/minFunc2012/minFunc/compiled']);
end

d = load(infile);
options = cdr_parseInputs({});
options.cidrePath = root;
[S, options] = cdr_loadImages(d.stack, options);
t = tic;
[model, options] = cdr_cidreModel(S, options); %#ok<ASGLU>
out.elapsed = toc(t);
out.S_pre = S;
out.v_small = model.v_small; out.z_small = model.z_small;
out.v = model.v; out.z = model.z;
out.lambda_v = options.lambda_vreg; out.bit_depth = options.bit_depth;
if is_octave
    save('-v7', outfile, '-struct', 'out');
else
    save(outfile, '-struct', 'out', '-v7');
end
end
