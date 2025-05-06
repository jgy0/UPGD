clc; clear; close all;

% 添加必要的工具路径
addpath('./tools/Metrics', './tools/Metrics/BRISQUE','./tools/Metrics/CCF','./tools/Metrics/CPBDM');
addpath('./tools/Metrics/DEFADE', './tools/Metrics/FADE','./tools/Metrics/ILNIQE','./tools/Metrics/LPC');
addpath('./tools/Metrics/NIQE', './tools/Metrics/SSEQ');

% 设置路径
ref_folder = 'D:\Users\Desktop\result\ref_result\gt';
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

% 获取参考图像列表
ref_files = dir(fullfile(ref_folder, '*.png'));
num_images = length(ref_files);
num_methods = length(method_names);

% 初始化结果存储
metrics = {'PSNR', 'SSIM'};
num_metrics = length(metrics);
all_results = zeros(num_methods, num_metrics); % 存储平均结果
detailed_psnr = cell(num_images+1, num_methods+1); % 存储详细PSNR结果
detailed_ssim = cell(num_images+1, num_methods+1); % 存储详细SSIM结果

% 初始化表头
detailed_psnr{1,1} = 'ImageName';
detailed_ssim{1,1} = 'ImageName';
for m = 1:num_methods
    detailed_psnr{1,m+1} = method_names{m};
    detailed_ssim{1,m+1} = method_names{m};
end

% 处理每个图像
for i = 1:num_images
    % 读取并处理参考图像
    ref = imread(fullfile(ref_folder, ref_files(i).name));
    ref = imresize(ref, [256, 256]);
    
    % 存储当前图像名
    detailed_psnr{i+1,1} = ref_files(i).name;
    detailed_ssim{i+1,1} = ref_files(i).name;
    
    % 对每个方法进行处理
    for m = 1:num_methods
        try
            % 获取方法文件夹中的对应图像
            % listing_jpg = dir(fullfile(images_dir, '*.jpg'));
            % listing_png = dir(fullfile(images_dir, '*.png'));
            % listing = [listing_jpg; listing_png];
            method_files_png = dir(fullfile(method_folders{m}, '*.png'));
            method_files_jpg = dir(fullfile(method_folders{m}, '*.jpg'));
            method_files=[method_files_jpg;method_files_png];
            % 读取并处理方法图像
            method_im = imread(fullfile(method_folders{m}, method_files(i).name));
            method_im = imresize(method_im, [256, 256]);
            
            % 计算指标
            output = metric_ref(method_im, ref);
            
            % 存储详细结果
            detailed_psnr{i+1, m+1} = output(2); % PSNR
            detailed_ssim{i+1, m+1} = output(3); % SSIM
            
            % 累加结果用于计算平均值
            all_results(m, 1) = all_results(m, 1) + output(2); % PSNR
            all_results(m, 2) = all_results(m, 2) + output(3); % SSIM
            
            % 显示进度
            fprintf('处理: %s - 图像 %d/%d - PSNR: %.2f, SSIM: %.4f\n', ...
                method_names{m}, i, num_images, output(2), output(3));
            
        catch ME
            warning(['处理 ' method_names{m} ' 的第 ' num2str(i) ' 张图像时出错: ' ME.message]);
            detailed_psnr{i+1, m+1} = NaN;
            detailed_ssim{i+1, m+1} = NaN;
        end
    end
end

% 计算平均结果
all_results = all_results / num_images;

% 创建并保存汇总表格
summary_table = array2table(all_results, ...
    'RowNames', method_names, ...
    'VariableNames', metrics);
writetable(summary_table, 'summary_results.csv', 'WriteRowNames', true);

% 保存详细结果
writecell(detailed_psnr, 'detailed_psnr_results.csv');
writecell(detailed_ssim, 'detailed_ssim_results.csv');

disp('所有处理完成！结果已保存到:');
disp('1. summary_results.csv - 包含平均PSNR和SSIM');
disp('2. detailed_psnr_results.csv - 包含每张图像的PSNR');
disp('3. detailed_ssim_results.csv - 包含每张图像的SSIM');