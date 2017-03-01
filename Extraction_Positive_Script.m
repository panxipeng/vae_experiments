% This script aims to automatically extract the nucleis from the trainning
% images, and pack them as an acceptable array for the Keras framework.
% ***Important Notice: The reshape() of Matlab follows the Fortran order, while Python's reshape() follows C order. 
Training_Num = 50;
Total_Nucleis = 0;
Patch_Radius = 8;
nucleis_data = zeros(1,867);

for i = 1:Training_Num
    
    Current_Path = ['img', num2str(i)];
    Img_File = [Current_Path, '/', 'img', num2str(i), '.bmp'];
    Mat_File = [Current_Path, '/', 'img', num2str(i), '_detection.mat'];
    img = imread(Img_File);
    load(Mat_File);
    
    for j = 1:length(detection)
        
        nuclei_boundary = [(detection(j,1)-8),(detection(j,1)+8);(detection(j,2)-8),(detection(j,2)+8)];
        if (min(min(nuclei_boundary)) <= 0 || max(max(nuclei_boundary)) > 500 )
            % neglect nucleis around the edge.
            continue;
        else
            img_nuclei = img(nuclei_boundary(2,1):nuclei_boundary(2,2),nuclei_boundary(1,1):nuclei_boundary(1,2),:);
            % reshape and store
            nucleis_data = [nucleis_data;reshape(img_nuclei,1,[])];
        end
        
    end
    
end

csvwrite('nucleis_data_original.dat',nucleis_data(2:end,:));
csvwrite('nucleis_data.dat',nucleis_data(2:10001,:));

% save workspace
save('train_pos_data_workspace')

% Show samples randomly 50*50 size

subsize = 10;
nucleis_data = nucleis_data(2:end,:);
figure
ha = tight_subplot(subsize,subsize,[.01 .01],[.01 .01],[.01 .01])
for k = 1:(subsize^2)
    axes(ha(k));
    item = round(10000*rand());
    imshow(reshape(nucleis_data(item,:),[17,17,3]))
end
