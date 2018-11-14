clc
clear
RAW_PATH = '/home/zhoujie/liveness detection/zjraw/non-face3/';

file =dir([RAW_PATH ,'*.raw']);

for num=1:length(file)
f1 = fopen([RAW_PATH,file(num).name], 'r');
data0 = fread(f1, 'uint16');
fclose(f1);
img1 = reshape(data0, 400, 345);
dep_img = img1';
imshow(dep_img,[350,800]); 
mouse=imrect; 
pos=getPosition(mouse)% x1 y1 w h 
pos=round(pos);


txtname = strrep(file(num).name,'.raw','.txt');
fp=fopen([RAW_PATH,txtname],'a');
fprintf(fp,'%i %i %i %i\n',pos); 
fclose(fp);

end

