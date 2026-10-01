% Collaborative State Transition Algorithm (CoSTA)
% ---------------------------------------------------------------------------

function [BestScore,BestPos,ConvergenceCurve] = CoSTA(N,MaxFES,lb,ub,dim,fobj)

[N,MaxFES,lb,ub] = validateInputs(N,MaxFES,lb,ub,dim);

% ---- parameters ----
mu        = max(2,floor(0.8*N));   % number of elites used in the covariance
Larch     = 3*N;                 % capacity of the successful-transition archive
lr        = 0.05;                % EMA learning rate of operator success rates
sigma0    = 1.0;                 % initial global step size
sigmaMin  = 1e-4; sigmaMax = 2;  % step-size bounds
nOp       = 4;                   % 1=expansion 2=rotation (with axesion) 3=experience translation 4=peer translation
opFloor   = 0.2;               % lower bound on operator-selection probability
stallLim  = 40;                  % generations of stagnation before a restart
restartFr = 0.2;                 % fraction of the worst individuals reinitialized


wRank = log(mu+0.5)-log(1:mu)'; wRank = wRank/sum(wRank);

% ---- initialize the population ----
X = lb+rand(N,dim).*(ub-lb);
Cost = zeros(N,1);
ConvergenceCurve = zeros(1,MaxFES);
FES = 0;
for i = 1:N
    Cost(i) = fobj(X(i,:));
    FES = FES+1;
    ConvergenceCurve(FES) = min(Cost(1:i));
end
[BestScore,bi] = min(Cost);
BestPos = X(bi,:);
ConvergenceCurve(1:FES) = BestScore;

% ---- archive and adaptive state ----
Adx = zeros(Larch,dim); Aw = zeros(Larch,1); aptr = 0; acount = 0;
opSucc = ones(1,nOp);
sigma  = sigma0;
stall  = 0;

while FES < MaxFES
    bestBefore = BestScore;
    [~,sortedIdx] = sort(Cost,'ascend');
    elites = sortedIdx(1:mu);

    % ---- elite weighted mean and covariance ----
    m = wRank'*X(elites,:);                 
    C = zeros(dim);
    for j = 1:mu
        dxj = X(elites(j),:)-m;
        C = C + wRank(j)*(dxj'*dxj);
    end
    C = (C+C')/2 + 1e-12*eye(dim);
    [Bm,D] = eig(C);
    dvec = sqrt(max(diag(D),0));      
    if all(dvec < eps)                      
        dvec = ones(dim,1)*mean(max(ub-lb,eps))*1e-3;
    end

    succCnt = 0; trialCnt = 0;
    opProb = (opSucc+opFloor)/sum(opSucc+opFloor);
    opCum  = cumsum(opProb);

    for i = 1:N
        if FES >= MaxFES, break; end
        xi = X(i,:);

        % Roulette selection of an operator from recent success rates
        op = find(rand <= opCum,1);

        switch op
            case 1  % expansion
                  
                ec = X(elites(randi(mu)),:);
                newPos = ec + sigma*(Bm*(dvec.*randn(dim,1)))';
            case 2  % rotation
                    
                newPos = xi + sigma*(Bm*(dvec.*randn(dim,1)))';
            case 3  % experience translation
                if acount > 0
                    k = rouletteIdx(Aw(1:acount));
                    newPos = xi + (0.5+rand)*Adx(k,:);
                else
                    newPos = xi + sigma*(Bm*(dvec.*randn(dim,1)))';
                end
            otherwise % peer translation
                betters = sortedIdx(1:max(1,find(sortedIdx==i,1)-1));
                if numel(betters) >= 1 && ~(numel(betters)==1 && betters(1)==i)
                    pb = betters(randi(numel(betters)));
                    newPos = xi + rand(1,dim).*(X(pb,:)-xi);
                else
                    newPos = m + sigma*(Bm*(dvec.*randn(dim,1)))';
                end
        end

        newPos = clip(newPos,lb,ub);
        newCost = fobj(newPos);
        FES = FES+1;
        trialCnt = trialCnt+1;

        improved = newCost < Cost(i);
        if improved
            gain = Cost(i)-newCost;
            % Store the successful displacement and its improvement weight
            aptr = mod(aptr,Larch)+1;
            Adx(aptr,:) = newPos-X(i,:);
            Aw(aptr)    = gain;
            acount = min(acount+1,Larch);

            Cost(i) = newCost; X(i,:) = newPos;
            succCnt = succCnt+1;
            if newCost < BestScore, BestScore = newCost; BestPos = newPos; end
        end
        opSucc(op) = (1-lr)*opSucc(op)+lr*double(improved);
        ConvergenceCurve(min(FES,MaxFES)) = BestScore;
    end

    % ---- one-fifth success rule for the global step size ----
    if trialCnt > 0
        ps = succCnt/trialCnt;
        if ps > 0.2, sigma = sigma*1.05; else, sigma = sigma*0.9; end
        sigma = min(max(sigma,sigmaMin),sigmaMax);
    end

    % ---- stagnation restart ----
    if BestScore < bestBefore-max(eps,1e-8*abs(bestBefore)), stall = 0; else, stall = stall+1; end
    if stall >= stallLim && FES < MaxFES
        nR = floor(restartFr*N);
        worstIdx = sortedIdx(end-nR+1:end);   
        for k = 1:numel(worstIdx)
            if FES >= MaxFES, break; end
            j = worstIdx(k);
            cand = lb+rand(1,dim).*(ub-lb);
            newCost = fobj(cand); FES = FES+1;
            X(j,:) = cand; Cost(j) = newCost;
            if newCost < BestScore, BestScore = newCost; BestPos = cand; end
            ConvergenceCurve(min(FES,MaxFES)) = BestScore;
        end
        sigma = sigma0; stall = 0;
    end
end

end

% =====================================================================
function k = rouletteIdx(wpos)
c = cumsum(wpos); tot = c(end);
if tot <= 0, k = randi(numel(wpos)); return; end
k = find(rand*tot <= c,1);
if isempty(k), k = numel(wpos); end
end

% ---------------------------------------------------------------------
function p = clip(p,lb,ub)
under = p < lb; over = p > ub;
if any(under), p(under) = lb(under)+rand(1,nnz(under)).*(ub(under)-lb(under)); end
if any(over),  p(over)  = lb(over) +rand(1,nnz(over)) .*(ub(over) -lb(over));  end
end

% ---------------------------------------------------------------------
function [N,MaxFES,lb,ub] = validateInputs(N,MaxFES,lb,ub,dim)
if N < 1 || MaxFES < 1 || dim < 1
    error('CoSTA:InvalidInput','N, MaxFES, and dim must be positive integers.');
end
N = floor(N); MaxFES = floor(MaxFES); N = min(N,MaxFES);
lb = reshape(lb,1,[]); ub = reshape(ub,1,[]);
if isscalar(lb), lb = repmat(lb,1,dim); end
if isscalar(ub), ub = repmat(ub,1,dim); end
if numel(lb) ~= dim || numel(ub) ~= dim || any(lb > ub)
    error('CoSTA:InvalidBounds','Each bound must have length dim, and lb must not exceed ub.');
end
end
