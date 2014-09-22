% Topo_anisotropy_correlogram: code to determine directional dependence of topography
% over different scales based on elevation.

% Interest is in continuous correlation, I don't want adjacent valleys to
% influence the signal and skew the semiminor axis. Ways to avoid this:
% Average correlation up to the max radius. Easy to do, problem is that
% solution is not devoted to a single wavelength, it is the mean up to a
% specific wavelength. (topo_anisotropy_avg)
% Terminate search at divides, this will require breaking into the .net and
% searching for points with no contributors. Not worth it...
% ignore it, just use the simple step search and take closest value per
% spoke
% Broad step search over wavelength interval per spoke. Makes for more
% robust solution, may be tricky to pull off.
% Terminate search at minimum correlation. Problem is it yields
% scale-dependent results. (this one)
tic
radius=10; radstep=5; % measure correlation to radius advancing by radstep
% radius=500; radwindow=2e4; radstep=2; % measure correlation to radius advancing by radstep
angle=((0:5:355)*pi/180)'; % 5 degree separation of spokes.
% dat=uint16(dat);
% cmean=mean(double(dat(:)));
% cor=uint32(zeros(length(angle),radius/radstep)); % initialize raw correlogram matrix FOR FULL AVERAGING
% dat=double(dat);
cor=zeros(length(angle),radius/radstep); % initialize raw correlogram matrix FOR FULL AVERAGING
cmatrix=cor;
% cor_bi=uint32(zeros(length(angle)/2,length(cor(1,:))));
cor_bi=zeros(length(angle)/2,length(cor(1,:)));
%mean_norm_cor=cor; % initialize mean normalized correlogram matrix
anisotropy=nan([length(geotiff_data(:,1)),length(geotiff_data(1,:)),radius]);
azimuth=anisotropy;

% xx=cursor_info.Position(1,1);
% yy=cursor_info.Position(1,2);
% cc=cursor_info.Position(1,3);
%
% if radius>xx || radius>yy || radius+xx>length(dat(1,:)) || radius+yy>length(dat(:,1))
%     error('Radius exceeds domain limits at prescribed coordinates. Stop thinking outside the box.');
% end

% cropdat=dat(yy-radius:yy+radius,xx-radius:xx+radius);
% % measure correlation per radius per angle, generates correlogram matrix
% WINDOW AVRAGING
for y=1:length(geotiff_data(:,1))%data by row
    if y>length(geotiff_data(:,1))-radius-1 || y<radius+1, continue, end
    for x=1:length(geotiff_data(1,:))%data by column
        if x>length(geotiff_data(1,:))-radius-1 || x<radius+1, continue, end
        for j=1:radstep:radius % Step through radii
            rad=j; % Advance radius length per iteration
            for i=1:length(angle) % Step through angles, 5 degree intervals
                xrad=cos(angle(i))*rad+x; % Find adjacent length (x component)
                yrad=sin(angle(i))*rad+y; % Find opposite length (y component)
                xrad=round(xrad);
                yrad=round(yrad);
                %             cor(i,j)=((cc-c(ind)).^2); % Measure correlation and stick it in an angle x radius matrix
                %             cor(i,j)=((cc-c(ind)).^2)./2; % Measure correlation and stick it in an angle x radius matrix
                cmatrix(i,j)=double(geotiff_data(yrad,xrad));
                %         cmean=mean(cmatrix(i,1:j));
                %             cor(i,j)=sum((cmatrix(i,1:j)-cmean).^2,2)./j;
                %Summing up by rows; Get the sum of the rows
                cor(i,j)=sum((cmatrix(i,1:j)-double(geotiff_data(y,x))).^2,2)./(2*j);
                %         cor(i,j)=(sum((1:1:j)-mean(1:1:j))*sum(cmatrix(i,1:j)-mean(cmatrix(i,1:j))))/sqrt(sum((1:1:j)-mean(1:1:j))^2*sum(cmatrix(i,1:j)-mean(cmatrix(i,1:j)))^2);
            end
           
            for k=1:length(cor(:,1))/2
                cor_bi(k,j)=mean([cor(k,j) cor(k+36,j)]);
                %gettin the mean of each column and storing 
                %it in cor_bi
            end
            
            [val1,ind]=min(cor_bi(:,j)); % How about just taking the 1-ratio of strongest correlation/orthogonal correlation value?
            if ind <= 18
              
                val2=cor_bi(ind+18,j);
            else
               
                val2=cor_bi(ind-18,j);
            end
           
            if val1==0
                if val2<1 && val2>0
                    val1=val2;
                else
                    val1=1;
                end
            end
            if val2==0, val2=1; end
            anisotropy(y,x,j)=val2/val1;
            azimuth(y,x,j)=rad2deg(angle(ind));
            
          
        end
    end
end
toc
save AN.mat anisotropy
save AZ.mat azimuth