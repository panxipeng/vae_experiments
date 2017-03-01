for i = 1:100
    
    Current_Path = ['img', num2str(i)];
    Mat_File = [Current_Path, '/', 'img', num2str(i), '_detection.mat'];
    load(Mat_File);
    csv_path = [Current_Path, '/', 'img', num2str(i), '_detection.csv'];
    csvwrite(csv_path, detection)
    
end