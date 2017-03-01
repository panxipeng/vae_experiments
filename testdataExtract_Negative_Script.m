% This script aims to extract negative samples from datasets for testing
% usage.

img_range = 50;
sampling_method = 2; % 1 sequential sampling. 2 random sampling
sampling_num = 500;
nucleis_neg_data = zeros(1,867);
sliding_window_center = [8 8];

for i = 1:img_range
    
    Current_Path = ['img', num2str(i)];
    Img_File = [Current_Path, '/', 'img', num2str(i), '.bmp'];
    Mat_File = [Current_Path, '/', 'img', num2str(i), '_detection.mat'];
    img = imread(Img_File);
    load(Mat_File);
    total_nucleis = size(detection,1);
%     figure(1);
%     imshow(img);
%     hold on;
%     scatter(detection(:,1),detection(:,2))
    if sampling_method == 2
        
        for j = 1:sampling_num
            sliding_window_center = (472 - 9).* rand(1,2,1) + 9;
            dist = sqrt((sliding_window_center(1) - detection(:,1)).^2 + (sliding_window_center(2) - detection(:,2)).^2);
            if any(dist<12)
                nucleis_neg_data = nucleis_neg_data;
            else
                nuclei_boundary = [(sliding_window_center(1)-8),(sliding_window_center(1)+8);(sliding_window_center(2)-8),(sliding_window_center(2)+8)];
%                 clf;
%                 imshow(img);
%                 hold on;
%                 scatter(detection(:,1),detection(:,2))
%                 rectangle('Position',[nuclei_boundary(1,1) nuclei_boundary(2,1) 17 17], 'EdgeColor','g', 'LineWidth',2)
                img_nuclei = img(nuclei_boundary(2,1):nuclei_boundary(2,2),nuclei_boundary(1,1):nuclei_boundary(1,2),:);
                nucleis_neg_data = [nucleis_neg_data;reshape(img_nuclei,1,[])];
            end
        end
    end
end

csvwrite('test_nucleis_neg_data_original.dat',nucleis_neg_data(2:end,:));
csvwrite('test_nucleis_neg_data.dat',nucleis_neg_data(2:10001,:));
% save workspace
save('test_neg_data_workspace')

% Show samples randomly 50*50 size

subsize = 10;
nucleis_neg_data = nucleis_neg_data(2:end,:);
figure
ha = tight_subplot(subsize,subsize,[.01 .01],[.01 .01],[.01 .01])
for k = 1:(subsize^2)
    axes(ha(k));
    item = round(10000*rand());
    imshow(reshape(nucleis_neg_data(item,:),[17,17,3]))
end