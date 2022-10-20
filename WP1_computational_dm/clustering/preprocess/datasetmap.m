function [Xmapped, kp] = datasetmap(Xm,varargin)
% Description: 
% Map input data set with missing values into normalized chains 
% around key points
%
% Function calls:
%      Xmapped = datasetmap(Xm)
%      Xmapped = datasetmap(Xm, distance)
%      Xmapped = datasetmap(Xm, distance, maxKey)
%
% Inputs:
%           Xm - Original data set with missing values  
%     distance - Selected distance metric. Default value: 'euc' 
%                Alternatives: 
%                'sqe' - squared Euclidean distance
%                'euc' - Euclidean distance
%                'cit' - City block distance 
%       maxKey - Maximum number of keypoints. Default value: 25 
% Output:
%      Xmapped - Mapped data set 
%
defaultDistance = 'euc';
defaultMaxKey = 25;
p = inputParser;
addOptional(p, 'distance', defaultDistance, ... 
                @(x) ismember(x,{'sqe','euc','cit'}));
addOptional(p, 'maxKey', defaultMaxKey, @(x) isnumeric(x));
parse(p,varargin{:});
distance = p.Results.distance;
maxKey = p.Results.maxKey;
%
if strcmp(distance,'sqe')
    dist = 'squaredeuclidean';
elseif strcmp(distance,'euc')
    dist = 'euclidean';
elseif strcmp(distance,'cit')
    dist = 'cityblock';
end  
%
%MDS
if size(Xm,2) > 2
    if ~isempty(find(isnan(Xm)))
        K = 5;
        Ximp = ICknnimpute(Xm,K,distance);
        opts = statset('Display','iter');
        X = mdscale(pdist(Ximp),2,'Start','random','Options',opts);
    else
        opts = statset('Display','iter');
        X = mdscale(pdist(Xm),2,'Start','random','Options',opts);
    end
else
    if ~isempty(find(isnan(Xm)))
        K = 1;
        X = knnimpute(Xm,K,distance);
    else
        X = Xm;
    end
end
fprintf('Performing mapping...\n');
%
%Density
density = zeros(size(X,1),1);
for i = 1:size(X,1)
    Xi = X(i,:);
    D = pdist2(Xi,X,dist);
    [~, idx] = sort(D);
    knearests = X(idx(2:5),:);
    for j = 1:size(knearests,1)
        density(i) = density(i) + pdist2(Xi,knearests(j,:),dist);
    end
    density(i) = 1/density(i);
end
%
%Key points
sigma = inf(size(X,1),1);
kp = inf(size(X,1),1);
for i = 1:size(X,1)
    Xi = X(i,:);
    D = pdist2(Xi,X,dist);
    [~, idx] = sort(D);
    for j = 2:length(idx)
        if density(i) < density(idx(j)) 
            sigma(i) = pdist2(Xi,X(idx(j),:),dist);
            break;
        end
    end
    kp(i) = density(i)*sigma(i);
end
nkey = min([round(sqrt(size(X,1))); maxKey]);
[~, idx] = sort(kp,'descend');
kp = idx(1:nkey);
%
% figure; 
% hold on; 
% scatter(X(:,1),X(:,2),'go');
% for i = 1:length(kp)
%     scatter(X(kp(i),1),X(kp(i),2),'ro','filled');
%     text(X(kp(i),1),X(kp(i),2),int2str(kp(i)));
% end
%
%Find pairs
N = size(X,1);
D = pdist2(X,X,dist);
pairs = zeros(N-nkey,2);
cnt = 1;
[~, idx1] = sort(density);
%idx1 = 1:N;
for i = 1:N
    Xi = X(idx1(i),:);
    Di = D(idx1(i),:);
    if ~ismember(idx1(i),kp)
        [~, idx] = sort(Di);
        for j = 1:length(idx)
            if density(idx1(i)) < density(idx(j))
                I = idx(j);
                break;
            end
        end
        pairs(cnt,:) = [idx1(i),I];
        cnt = cnt + 1;
    end
end
%
%Form chains
chains = cell(nkey,1);
chainCnt = ones(nkey,1);
startpoints = zeros(size(pairs,1),1);
cnt = 1;
[~, idx] = sort(density);
%idx = 1:N;
for i = 1:N
    chain = [];
    if ~ismember(idx(i),startpoints) && ~ismember(idx(i),kp)
        startpoints(cnt) = idx(i);
        cnt = cnt + 1;
        chain = [chain; idx(i)];
        I = find(pairs(:,1)==idx(i));
        endp = pairs(I,2);
        while ~ismember(endp,kp)
            if ~ismember(endp,startpoints)
                startpoints(cnt) = endp;
                cnt = cnt + 1;
            end
            chain = [chain; endp];
            I = find(pairs(:,1)==endp);
            endp = pairs(I,2);
        end
        chain = [chain; endp];
        I = find(endp==kp);
        chains{I,chainCnt(I)} = flipud(chain);
        chainCnt(I) = chainCnt(I) + 1;
    end
end
chainCnt = chainCnt - 1;
%
%Compute sizes of clusters
Tkp = zeros(nkey,1);
Tkm = zeros(size(chains));
for i = 1:nkey
    Tkp1 = [];
    for j = 1:chainCnt(i)
        A = chains{i,j};
        Tkp1 = [Tkp1; A];
        for k = 1:length(A)-1
            Tkm(i,j) = Tkm(i,j) + pdist2(X(A(k),:),X(A(k+1),:),dist);
        end
    end
    Tkp(i) = length(unique(Tkp1));
end
Tkm(Tkm==0) = NaN;
Tkm = median(Tkm,2,'omitnan');
%
%Remove small and interconnected clusters 
thres = round(sum(Tkp)/(2*nkey));
idx = find(Tkp<=thres);
[~, idx2] = sort(density(kp),'descend');
I = find(idx2(1)==idx);
if ~isempty(I)
    idx(I) = [];
end
kp(idx) = [];
nkey = nkey - length(idx);
Tkp(idx) = [];
Tkm(idx) = [];
%
thres2 = sum(Tkm)/(1.5*nkey);
D = pdist2(X(kp,:),X(kp,:),dist);
D1 = D;
D1(D<=thres2) = 1;
D1(D>thres2) = 0;
D1 = triu(D1,1);
idx = cell(nkey,1);
for i = 1:size(D1,1)
    Di = D1(i,:);
    I = find(Di==1);
    if ~isempty(I)
        idx{i,1} = [i; I'];
    end
end
for i = 1:nkey-1
    A = idx{i,1};
    flag = 1;
    while flag == 1
        flag = 0;
        for j = i+1:nkey
            B = idx{j,1};
            if ~isempty(intersect(A,B))
                idx{i,1} = union(A,B);
                idx{j,1} = [];
                flag = 1;
            end
        end
    end
end
idx3 = [];
for i = 1:nkey
    A = idx{i,1};
    if ~isempty(A)
        [~, idx2] = sort(density(kp(A)),'descend');
        idx3 = [idx3; A(idx2(2:end))];
        X(kp(A(idx2(1))),:) = nanspatialmedianfunc(X(kp(A),:));
        density(kp(A(idx2(1)))) = Inf;
    end
end
idx3 = unique(idx3);
kp(idx3) = [];
nkey = nkey - length(idx3);
Tkp(idx3) = [];
Tkm(idx3) = [];
%
% figure;
% hold on;
% scatter(X(:,1),X(:,2),'go');
% for i = 1:length(kp)
%     scatter(X(kp(i),1),X(kp(i),2),'ro','filled');
%     text(X(kp(i),1),X(kp(i),2),int2str(kp(i)));
% end
%
%Recompute pairs
N = size(X,1);
D = pdist2(X,X,dist);
pairs = zeros(N-nkey,2);
cnt = 1;
[~, idx1] = sort(density);
%idx1 = 1:N;
for i = 1:N
    Xi = X(idx1(i),:);
    Di = D(idx1(i),:);
    if ~ismember(idx1(i),kp)
        [~, idx] = sort(Di);
        for j = 1:length(idx)
            if density(idx1(i)) < density(idx(j))
                I = idx(j);
                break;
            end
        end
        pairs(cnt,:) = [idx1(i),I];
        cnt = cnt + 1;
    end
end
%
%Recompute chains
chains = cell(nkey,1);
chainCnt = ones(nkey,1);
startpoints = zeros(size(pairs,1),1);
cnt = 1;
[~, idx] = sort(density);
%idx = 1:N;
for i = 1:N
    chain = [];
    if ~ismember(idx(i),startpoints) && ~ismember(idx(i),kp)
        startpoints(cnt) = idx(i);
        cnt = cnt + 1;
        chain = [chain; idx(i)];
        I = find(pairs(:,1)==idx(i));
        endp = pairs(I,2);
        while ~ismember(endp,kp)
            if ~ismember(endp,startpoints)
                startpoints(cnt) = endp;
                cnt = cnt + 1;
            end
            chain = [chain; endp];
            I = find(pairs(:,1)==endp);
            endp = pairs(I,2);
        end
        chain = [chain; endp];
        I = find(endp==kp);
        chains{I,chainCnt(I)} = flipud(chain);
        chainCnt(I) = chainCnt(I) + 1;
    end
end
chainCnt = chainCnt - 1;
%
%Remove intersections of chains
chains2 = chains;
for i = 1:nkey
    for j = 1:chainCnt(i) 
        A = chains{i,j};
        for k = j+1:chainCnt(i)
            B = chains2{i,k};
            C = intersect(A,B);
            I = 0;
            if length(C)>1
                for l = 1:length(C)
                    I1 = find(C(l)==A);
                    if I1 > I
                        I = I1;
                    end
                end
                I2 = find(A(I)==B);
                chains2{i,k} = B(I2:end);
            end
        end
    end
end
%
%Compute lengths of subchains
Tkc = zeros(size(chains));
% Tkn = zeros(nkey,1);
Tkm = zeros(size(chains));
for i = 1:nkey
    for j = 1:chainCnt(i)
        A = chains2{i,j};
        B = chains{i,j};
        for k = 1:length(A)-1
            Tkc(i,j) = Tkc(i,j) + pdist2(X(A(k),:),X(A(k+1),:),dist);
            %Tkn(i) = Tkn(i) + 1;
        end
        for k = 1:length(B)-1
            Tkm(i,j) = Tkm(i,j) + pdist2(X(B(k),:),X(B(k+1),:),dist);
        end
    end
end
% sum(Tkn)
Tkm(Tkm==0) = NaN;
Tkm = median(Tkm,2,'omitnan');
%
%Normalization
X1 = X;
divisor = zeros(size(X,1),1);
for i = 1:nkey
    for j = 1:chainCnt(i) 
        A = chains{i,j};
        B = chains2{i,j};
        X2 = X;
        I = find(B(1)==A);
        if I == 1
            divisor(B) = Tkc(i,j);
        end
        Tkc2 = 0;
        for k = 1:I-1
            startp = X2(A(k+1),:);
            endp = X2(A(k),:);
            d = pdist2(startp,endp,dist);
            u = (endp - startp)/d;
            X2(A(k+1),:) = startp + (d-d/divisor(A(k+1)))*u;
            for l = k+2:size(A,1)
                startp = X2(A(l),:);
                X2(A(l),:) = startp + (d-d/divisor(A(k+1)))*u;
            end
            Tkc2 = Tkc2 + pdist2(X1(A(k),:),X1(A(k+1),:),dist); 
        end
        for k = 1:size(B,1)-1
            startp = X2(B(k+1),:);
            endp = X2(B(k),:);
            d = pdist2(startp,endp,dist);
            u = (endp - startp)/d;
            if divisor(B(k+1)) == 0
                divisor(B(k+1)) = Tkc(i,j)/(1-Tkc2);  
            end
            X2(B(k+1),:) = startp + (d-d/divisor(B(k+1)))*u;
            X1(B(k+1),:) = startp + (d-d/divisor(B(k+1)))*u;
            for l = k+2:size(B,1)
                startp = X2(B(l),:);
                X2(B(l),:) = startp + (d-d/divisor(B(k+1)))*u;
                X1(B(l),:) = startp + (d-d/divisor(B(k+1)))*u;
            end
        end
    end
end
%
%Scale normalized chains
Xorg = X;
X = X1;
%Tkm = ones(nkey,1)*median(Tkm);
for i = 1:nkey
    for j = 1:chainCnt(i) 
        A = chains{i,j};
        B = chains2{i,j};
        X2 = X;
        I = find(B(1)==A);
        Tkc2 = 0;
        for k = 1:I-1
            startp = X2(A(k+1),:);
            endp = X2(A(k),:);
            d = pdist2(startp,endp,dist);
            u = (endp - startp)/d;
            X2(A(k+1),:) = startp + (d-d*Tkm(i))*u;
            for l = k+2:size(A,1)
                startp = X2(A(l),:);
                X2(A(l),:) = startp + (d-d*Tkm(i))*u;
            end
            Tkc2 = Tkc2 + pdist2(X1(A(k),:),X1(A(k+1),:),dist); 
        end
        for k = 1:size(B,1)-1
            startp = X2(B(k+1),:);
            endp = X2(B(k),:);
            d = pdist2(startp,endp,dist);
            u = (endp - startp)/d;
            X2(B(k+1),:) = startp + (d-d*Tkm(i))*u;
            X1(B(k+1),:) = startp + (d-d*Tkm(i))*u;
            for l = k+2:size(B,1)
                startp = X2(B(l),:);
                X2(B(l),:) = startp + (d-d*Tkm(i))*u;
                X1(B(l),:) = startp + (d-d*Tkm(i))*u;
            end
        end
    end
end
X = Xorg;
%
% chainLen = zeros(size(chains));
% for i = 1:nkey
%     for j = 1:chainCnt(i) 
%         A = chains{i,j};
%         for k = 1:length(A)-1
%             chainLen(i,j) = chainLen(i,j) + pdist2(X1(A(k),:),X1(A(k+1),:),dist);
%         end
%     end
% end
%
% fprintf('Visualizing the results...\n');
% figure;
% hold on;
% for i = 1:nkey
%     colors = lines(nkey);
%     for j = 1:chainCnt(i) 
%         A = chains2{i,j};
%         A = flipud(A);
%         xPoints = X(A,1);
%         yPoints = X(A,2);
%         l1 = 1:length(A)-1;
%         l2 = 2:length(A);
%         d = X(A(l2),:)-X(A(l1),:);
%         d = [d; 0, 0];
%         scatter(xPoints,yPoints,'o','CData',colors(i,:));
%         text(xPoints,yPoints,string(A));
%         quiver(xPoints,yPoints,d(:,1),d(:,2),'color',colors(i,:));             
%     end
% end
% scatter(X(kp,1),X(kp,2),'ro','filled');
%
% figure;
% hold on;
% for i = 1:nkey
%     colors = lines(nkey);
%     for j = 1:chainCnt(i) 
%         A = chains2{i,j};
%         A = flipud(A);
%         xPoints = X1(A,1);
%         yPoints = X1(A,2);
%         l1 = 1:length(A)-1;
%         l2 = 2:length(A);
%         d = X1(A(l2),:)-X1(A(l1),:);
%         d = [d; 0, 0];
%         scatter(xPoints,yPoints,'o','CData',colors(i,:));
%         text(xPoints,yPoints,string(A));
%         quiver(xPoints,yPoints,d(:,1),d(:,2),'color',colors(i,:));             
%     end
% end
% scatter(X1(kp,1),X1(kp,2),'ro','filled');
%
% figure;
% hold on;
% xlim([-1, 1]);
% ylim([-1, 1]);
% scatter(X(:,1),X(:,2),'bo');
% scatter(X(kp,1),X(kp,2),'ro','filled');
% % 
% figure;
% hold on;
% xlim([-1, 1]);
% ylim([-1, 1]);
% scatter(X1(:,1),X1(:,2),'bo');
% scatter(X1(kp,1),X1(kp,2),'ro','filled');
%
Xmapped = X1;

end

function C = nanspatialmedianfunc(X, varargin)
% Description: 
% Estimate spatial median value of input data set. Algorithm uses 
% available data strategy for treating missing values.  
%
% Function calls:
%         C = nanspatialmedianfunc(X)
%         C = nanspatialmedianfunc(X, max_iter)
%         C = nanspatialmedianfunc(X, max_iter, tol)  
%
% Inputs: 
%         X - Input data set
%  max_iter - Maximum number of iterations 
%             Default value: 100 
%       tol - Stopping criterion 
%             Default value: 1e-5 
%
% Output: 
%         C - Spatial median value of data set 
%
defaultMaxIter = 100;
defaultTol = 1e-5; 
%
p = inputParser;
addOptional(p, 'max_iter', defaultMaxIter, @(x) isnumeric(x));
addOptional(p, 'tol', defaultTol, @(x) isnumeric(x));
parse(p,varargin{:});
max_iter = p.Results.max_iter;
tol = p.Results.tol;
%
u = zeros(1,size(X,2));
for i = 1:size(X,2)
    I = ~isnan(X(:,i));
    u(i) = median(X(I,i));
end
P = ~isnan(X);
iters = 0;
w = 1.5;
while iters < max_iter
    iters = iters + 1;
    D = sqrt(nansum(bsxfun(@minus,X,u).^2,2));
    a = 1./(D+sqrt(sqrt(eps)));
    ax = nansum(bsxfun(@times,a,X));
    a = bsxfun(@times,P,a);
    v = (1./sum(a)).*ax;
    u1 = u + w*(v-u);
    if norm(u1-u,inf) < tol
        break;
    end
    u = u1;
end
C = u1;

end




