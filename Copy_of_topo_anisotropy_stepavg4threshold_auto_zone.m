% Topo_anisotropy_correlogram: code to determine directional dependence of topography
% over different scales based on elevation (or any spatial variable).

% Important outputs are aspect_ratio and tilt, or the anisotropy and
% direction of a prevailing fabric apparent at any scale between node
% spacing and domain limits.

% The computations are simple but expect millions of them. each point will
% have at least 36 computations and this number increases geometrically at
% higher scales.

% Options:
%   Window=0, continuous correlation, values averaged iteratively to create
%   a statistical expectation operator used to temper the aspect ratio
%   value (and indirectly, tilt)
%   Window=1, correlation dependent on points within a radius window,
%   define the radius and the radwindow

% Calculation is currently limited to 5 degree separation of spokes.
% reducing this will drastically increase computation time.
window=0;
radius=1800; radwindow=4000; radstep=900; % measure correlation to radius advancing by radstep
angle=((0:5:355)*pi/180)'; % 5 degree separation of spokes.
spoke=angle.*0; % initialize spoke array
tilt=[];
aspect_ratio=[];
coords=[];

% Assign dimensions of study area, try to keep it small. Necessary for huge
% maps. Comment out otherwise.
dat=fulldat;
left=9.892e5;
right=1.191e6;
up=3.011e6;
down=2.85e6;
dat=dat(dat(:,1)>=left-radius,:);
dat=dat(dat(:,1)<=right+radius,:);
dat=dat(dat(:,2)>=down-radius,:);
dat=dat(dat(:,2)<=up+radius,:);

% chord=zeros(length(angle)/2,2); % initialize chord array
if window==1
    cor=zeros(length(angle),radius/radstep-((radius-radwindow)/radstep)); % initialize raw correlogram matrix
else
    cor=zeros(length(angle),radius/radstep); % initialize raw correlogram matrix FOR FULL AVERAGING
end
cor_bi=zeros(length(angle)/2,length(cor(1,:)));
mean_norm_cor=cor; % initialize mean normalized correlogram matrix

x=dat(:,1); %topo X coords
y=dat(:,2); %topo Y coords
%c=round(dat(:,3));  %topo Z, rounded to integer (important for fine anisotropy in almost flat regions)
c=dat(:,3);
kk_prime=5; kk=0;
l=0;
%for k=1:1:length(dat)
for k = 556:755
    if k/length(dat)*100 >= kk
        fprintf('%f percent done\n',kk);
        kk=kk+kk_prime;
    end
    xx=dat(k,1);
    yy=dat(k,2);
    cc=dat(k,3);
    if radius>xx-min(x) || radius>yy-min(y) || radius+xx>max(x) || radius+yy>max(y)
        continue
%     elseif y(k)<2885e3 || y(k)>3108e3 || x(k)<9928e2 || x(k)>1215e3
%         continue
    end
   
    l=l+1;
    if window==1 % Window switch
        % % measure correlation per radius per angle, generates correlogram matrix
        % WINDOW AVRAGING
        for j=1:radius/radstep-((radius-radwindow)/radstep) % Step through radii
            rad=(radius-radwindow)+radstep*j; % Advance radius length per iteration
            for i=1:length(angle) % Step through angles, 5 degree intervals
                xrad=cos(angle(i))*rad; % Find adjacent length (x component)
                yrad=sin(angle(i))*rad; % Find opposite length (y component)
                dx=(xx+xrad)-x(:);
                dy=(yy+yrad)-y(:);
                dist=sqrt(dx.^2+dy.^2);
                [val,ind]=min(dist); % Find point closest to the radius value j for each angle i
                cor(i,j)=((cc-c(ind)).^2)*.5; % Measure correlation and stick it in an angle x radius matrix
                %         cor(i,j)=((cc-c(ind)))*.5; % Measure correlation and stick it in an angle x radius matrix
                %         cor(i,j)=abs(cc-c(ind));
            end
            if j>1
                cor(:,j)=mean(cor(:,1:j),2);
            end
        end
    else % For total averaging over radius iterations
        for j=1:radius/radstep % Step through radii, ALL AVERAGING
            rad=radstep*j; % Advance radius length per iteration
            for i=1:length(angle) % Step through angles, 5 degree intervals
                xrad=cos(angle(i))*rad; % Find adjacent length (x component)
                yrad=sin(angle(i))*rad; % Find opposite length (y component)
                dx=(xx+xrad)-x(:);
                dy=(yy+yrad)-y(:);
                dist=sqrt(dx.^2+dy.^2);
                [val,ind]=min(dist); % Find point closest to the radius value j for each angle i
                %fprintf('Val and ind are: \n');
                tempSub = cc - c(ind);
                tempPower = tempSub.^2;
                tempMult = tempPower * 0.5;
                cor(i,j)=((cc-c(ind))^2)*.5; % Measure correlation and stick it in an angle x radius matrix
                %         cor(i,j)=cc-c(ind);
                
                
            end
            if j>1
                cor(:,j)=mean(cor(:,1:j),2);
            end
        end
    end
    
    for j=1:length(cor(:,1))/2
        cor_bi(j,:)=(cor(j,:)+cor(j+36,:))./2;
    end
    
    % for k=2:length(cor_bi(1,:))
    %     cor_bi(:,k)=mean(cor_bi(:,1:k));
    % end
    val1=[]; ind=[]; val2=[]; semiminor=[];
    [val1,ind]=min(cor_bi(:,1:end)); % How about just taking the 1-ratio of strongest correlation/orthogonal correlation value?
    for j=1:length(ind)
        if ind(j) <= 18
            %     dist2=(val1-cor_bi(ind+18,:)).^2;
            %     [val2,ind2]=min(dist2); %Don't do this, instead search from closest to furthest, find first time when value gets close
            val2(j)=cor_bi(ind(j)+18,end);
        else
            %     dist2=(val1-cor_bi(ind-18,:)).^2;
            %     [val2,ind2]=min(dist2);
            val2(j)=cor_bi(ind(j)-18,end); %try this to take 1-ratio of strongest cor and orthogonal cor
        end
        % for i=1:length(dist2)
        % [val2,ind2]=min(dist2);
    end
    semimajor=radius;
    % semiminor=radstep*ind2;
    semiminor=radius*(val1./val2);
    aspect_ratio(l,:)=1-sqrt(semiminor.^2./semimajor^2);
    tilt(l,:)=rad2deg(angle(ind));
    coords(l,:)=[xx yy cc];
    fprintf('Hello\n');
    % Convert vals to degrees for ellipse build (recalc back later):
%     rad_deg=distdim(radius,'m','deg');
%     xx_deg=distdim(xx,'m','deg');
%     yy_deg=distdim(yy,'m','deg');
    
    % Create and position the ellipse based on aspect ratio, semimajor axis,
    % and source point coordinates
    % [elat,elon]=ellipse1(yy_deg,xx_deg,[rad_deg aspect_ratio],90-chord(ind,1));
%     [elat,elon]=ellipse1(yy_deg,xx_deg,[rad_deg aspect_ratio(l)],90-tilt(l));
%     e_y=distdim(elat,'deg','m'); % recalc back to meters
%     e_x=distdim(elon,'deg','m');
%     n=zeros(length(e_y))+(max(c)+10);
    
    % plot the ellipse, make it so it's transposed on the topo map.
    % figure;
%     hold on;
%     if window==1
%         plot3(e_x,e_y,n,'y','LineWidth',2); %need to figure out how to convert degrees to meters for this special case. Ellipse points are in degrees, coordinates are in meters.
%     else
%         plot3(e_x,e_y,n,'b','LineWidth',2); %need to figure out how to convert degrees to meters for this special case. Ellipse points are in degrees, coordinates are in meters.
%     end
% %     axis([min(x) max(x) min(y) max(y)]); % Here in case you aren't plotting over the topo map
end
