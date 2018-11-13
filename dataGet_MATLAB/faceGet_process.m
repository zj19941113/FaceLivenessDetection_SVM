
clear
RAW_PATH = '/home/zhoujie/liveness detection/zjraw/non-face3/';
jpg_path = '/home/zhoujie/liveness detection/svm/data/no-face/';

file =dir([RAW_PATH ,'*.raw']);
for j=1:length(file)
    try
        f1 = fopen([RAW_PATH,file(j).name], 'r');
        txtname = strrep(file(j).name,'.raw','.txt');
        [par1,par2,par3,par4] = textread([RAW_PATH,txtname],'%d%d%d%d',1);
        data0 = fread(f1, 'uint16');
        fclose(f1);
        img1 = reshape(data0, 400, 345);
        dep_img = img1';
        % dep_img(find(dep_img > 600))= 0;
        % figure(1),imshow(dep_img,[400,580]); 
        try
            face = dep_img(par2 :par2 +par4-2,par1 :par1 + par3-2);
        catch
             if par2 +par4-2 >345
                 face_height = 345;
             else
                 face_height = par2 +par4-2;
             end
             if par1 +par3-2 >400
                face_weight = 400;
             else
                face_weight = par1 +par3-2;
             end
             face = dep_img(par2 :face_height,par1 :face_weight);
        end
        
            % figure(2),imshow(face,[420,500]);
            [m,n]=size(face);
            faceData = reshape(face, 1, m*n);
            faceData(find(faceData==0))=[];
            able = 0;total = 0;
            for i =1:1000
                num = randperm(length(faceData),1);
                facePlane = faceData(num);
                distance = abs([-1,faceData(num)]*[faceData;ones(1,length(faceData))]);
                total=sum(distance<30); 
                if total>able           
                   able=total;
                   bestfacePlane=facePlane;
                end
            end
            % figure(3),imshow(face,[bestfacePlane-50,bestfacePlane+50]);
            xmax = bestfacePlane+50;
            xmin = bestfacePlane-50;
            face(find(face > xmax ))=xmax; 
            face(find(face < xmin ))=xmin; 
        %     figure(5),imshow(face,[bestfacePlane-50,bestfacePlane+50]);
            ymax=255;ymin=0; 
            OutImg = round((ymax-ymin)*(face-xmin)/(xmax-xmin) + ymin); %归一化并取整
            Outface=uint8(OutImg); 
        %     figure(4),imshow(Outface);
            Outface = imresize(Outface, [40 40]);
        %     figure(5),imshow(Outface);
            jpgname = [jpg_path,num2str(j+538),'.jpg'];
            imwrite(Outface,jpgname);
        
    catch
        disp(file(j).name)
    end

end

