clear;

envCfg = coder.gpuEnvConfig('host');
envCfg.DeepLibTarget = 'cudnn';
envCfg.DeepCodegen = 1;
envCfg.Quiet = 1;
coder.checkGpuInstall(envCfg);


% load('PedNet.mat');
% PedNet.Layers
% 
% im = imread('test.jpg');
% im = imresize(im, [480, 640]);
% 
% cfg = coder.gpuConfig('mex');
% cfg.TargetLang = 'C++';
% cfg.DeepLearningConfig = coder.DeepLearningConfig('cudnn')
% codegen -config cfg pedDetect_predict -args {im} -report