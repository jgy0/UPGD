clc;clear;close all;
addpath('./tools/Metrics', './tools/Metrics/BRISQUE','./tools/Metrics/CCF','./tools/Metrics/CPBDM');
addpath('./tools/Metrics/DEFADE', './tools/Metrics/FADE','./tools/Metrics/ILNIQE','./tools/Metrics/LPC');
addpath('./tools/Metrics/NIQE', './tools/Metrics/SSEQ');

str = '';

ref_folder = 'D:\Users\Desktop\result\ref_result\gt0-50';
method_folder = 'D:\Users\Desktop\tmp\18';
file_list = dir(fullfile([method_folder, '/', '*.png']));


    method_name={'real_underwater1'};
temp = cell(length(method_name),1);

fid = fopen(['real_underwater1',str,'.csv'],'w');
fprintf(fid,'%s,%s,%s,%s,%s,%s','Name',...
    'MSE','PSNR','SSIM');
fprintf(fid,'\n');

for i = 1 : 1 :length(file_list)
	ref=imread([ref_folder,'/',file_list(i).name]);
    ref= imresize(ref, [256, 256]);
    for j = 1:1:length(method_name) 
        method_image= imread([method_folder, '/', file_list(i).name]);
        method_image= imresize(method_image, [256, 256]);
        % 这里imshow是人为检查参考图像和增强图像是不是对上了，如果没对上要及时暂停
        imshow([method_image ref]);	
        output = metric_ref(method_image, ref);  
        temp{j,1}(i,1:3) = output;
        fprintf(fid,'%s,%s,%s,%s',file_list(i).name,...
            num2str(output(1)),num2str(output(2)),num2str(output(3))); 
    
    end
	fprintf(fid,'\n'); 
	fprintf('%d/%d\n', i, length(file_list));
end
fclose(fid);
matname = [str,'.mat'];
save('matname','temp');

