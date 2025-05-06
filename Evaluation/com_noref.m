clc; clear; close all;
addpath('./tools/Metrics', './tools/Metrics/BRISQUE','./tools/Metrics/CCF','./tools/Metrics/CPBDM');
addpath('./tools/Metrics/DEFADE', './tools/Metrics/FADE','./tools/Metrics/ILNIQE','./tools/Metrics/LPC');
addpath('./tools/Metrics/NIQE', './tools/Metrics/SSEQ');

% 设置路径和对比方法
method_folders = {
    'E:\compararative_Ex\CBLA\indoor\', 
    'E:\compararative_Ex\CCMSRNet\indoor\',
    'E:\compararative_Ex\DDformer\indoor\',
    'E:\compararative_Ex\DiffWater\indoor\',
    'E:\compararative_Ex\DM\indoor\',
    'E:\compararative_Ex\HCLP\indoor\',
    'E:\compararative_Ex\HLRP\indoor\',
    'E:\compararative_Ex\ICSP\indoor\',
   
    'E:\compararative_Ex\PUIE\indoor\',
    'E:\compararative_Ex\Semi-UIR\indoor\',
    'E:\compararative_Ex\ucolor\indoor\',
    'E:\compararative_Ex\UNTV\indoor\',
    'E:\compararative_Ex\ushape\indoor\',
    'E:\compararative_Ex\WWPF\indoor\'
};

method_names = {
    'CBLA', 'CCMSRNet', 'DDformer', 'DiffWater', 'DM',  'HCLP', 'HLRP', 'ICSP', 'PUIE',  'Semi-UIR', 'ucolor', 'UNTV', 'ushape', 'WWPF'
};


% 初始化参数
num_images = 51;  % 每个方法的图像数量
metrics = {'UIQM', 'UCIQE', 'NIQE', 'EME'};
num_methods = length(method_names);
num_metrics = length(metrics);

% 预分配结果矩阵
all_results = zeros(num_images, num_metrics, num_methods);  % 图像×指标×方法

% 对每个方法进行评估
for m = 1:num_methods
    fprintf('正在评估方法: %s (%d/%d)\n', method_names{m}, m, num_methods);
    
    % 获取当前文件夹中所有图像（png和jpg）
    img_files = [dir(fullfile(method_folders{m}, '*.png')); dir(fullfile(method_folders{m}, '*.jpg'))];
    
    % 检查图像数量
    if length(img_files) ~= num_images
        error('方法 %s 的图像数量不是51张（实际%d张）', method_names{m}, length(img_files));
    end
    
    % 按文件列表顺序评估
    for i = 1:num_images
        try
            img = imread(fullfile(method_folders{m}, img_files(i).name));
            all_results(i, :, m) = metric(img);
            
            % 显示进度
            if mod(i, 10) == 0
                fprintf('  已完成图像: %d/%d\n', i, num_images);
            end
        catch ME
            warning('处理 %s 的第 %d 张图像(%s)时出错: %s', ...
                   method_names{m}, i, img_files(i).name, ME.message);
            all_results(i, :, m) = NaN;
        end
    end
end

% 计算平均结果（忽略NaN值）
avg_results = squeeze(nanmean(all_results, 1))';  % 方法×指标

% 保存结果到CSV
fid = fopen('comparison_results.csv', 'w');
fprintf(fid, 'Method,UIQM,UCIQE,NIQE,EME\n');
for m = 1:num_methods
    fprintf(fid, '%s,%.4f,%.4f,%.4f,%.4f\n', ...
        method_names{m}, avg_results(m, 1), avg_results(m, 2), ...
        avg_results(m, 3), avg_results(m, 4));
end
fclose(fid);

% 保存详细结果到MAT文件
save('comparison_results.mat', 'all_results', 'avg_results', 'method_names');

% 绘制对比图
figure('Position', [100, 100, 800, 600]);
for k = 1:4
    subplot(2,2,k);
    bar(avg_results(:,k));
    title([metrics{k} '对比']); 
    ylabel(metrics{k});
    set(gca, 'XTickLabel', method_names, 'XTickLabelRotation', 45);
end
saveas(gcf, 'metrics_comparison.png');
disp('评估完成！结果已保存');