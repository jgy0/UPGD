clc;clear;close all;
addpath('./tools/Metrics', './tools/Metrics/BRISQUE','./tools/Metrics/CCF','./tools/Metrics/CPBDM');
addpath('./tools/Metrics/DEFADE', './tools/Metrics/FADE','./tools/Metrics/ILNIQE','./tools/Metrics/LPC');
addpath('./tools/Metrics/NIQE', './tools/Metrics/SSEQ');

str = '';
% orginal_folder ='D:\Users\Desktop\result\UCDP\PPM-u2';

method_folder = 'D:\Users\Desktop\result\ref_result\PPM-U2';
% file_list = dir(fullfile([orginal_folder, '/', '*.png']));
file_list1 = dir(fullfile([method_folder, '/', '*.png']));

method_name={'real_underwater1'};
temp = cell(length(method_name),1);

fid = fopen(['real_underwater1',str,'.csv'],'w');

fprintf(fid,'%s,%s,%s,%s,%s\n','Name',...
    'UIQM','UCIQE', 'NIQE', 'EME');
 fprintf(fid,'\n');
for i = 1 : 1 :length(file_list1)
    for j = 1:1:length(method_name) 
        method_image= imread([method_folder, '/', file_list1(i).name]);
        output = metric(method_image);  
        temp{j,1}(i,1:4) = output;
        fprintf(fid,'%s,%s,%s,%s,%s',file_list1(i).name,...
            num2str(output(1)),num2str(output(2)),num2str(output(3)),num2str(output(4))); 
    end
	fprintf(fid,'\n'); 
	fprintf('%d/%d\n', i, length(file_list1));
end
fclose(fid);
matname = [str,'.mat'];
save('matname','temp');

% % fid2 = fopen(['Avg_WF-Diff_Pol',str,'.csv'],'w');
% % fprintf(fid2,'%s,%s,%s,%s,%s,%s,%s\n','method',...
% %    'PSNR','SSIM','MSE');
% % % fprintf(fid2,'%s,%s,%s,%s,%s\n','method',...
% % %     'UIQM','UCIQE', 'NIQE','EME');
% % 
% % for k=1:length(method_name)
% %     mean_method = mean(temp{k,1});
% %     %fprintf(fid2,'%s,%s,%s,%s,%s,%s,%s\n',method_name{k},num2str(mean_method(1)),num2str(mean_method(2)),num2str(mean_method(3)),num2str(mean_method(4)),num2str(output(5)),num2str(output(6)));
% %     fprintf(fid2,'%s,%s,%s,%s,%s\n',method_name{k},num2str(mean_method(1)),num2str(mean_method(2)),num2str(mean_method(3)),num2str(mean_method(4)));
% % 
% % %     fprintf(fid2,'%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n',method_name{k},num2str(mean_method(1)),num2str(mean_method(2)),num2str(mean_method(3)),num2str(mean_method(4)),...
% % %        num2str(mean_method(5)),num2str(mean_method(6)),num2str(mean_method(7)),num2str(mean_method(8)), num2str(mean_method(9))); 
% % end
% % fclose(fid2);
