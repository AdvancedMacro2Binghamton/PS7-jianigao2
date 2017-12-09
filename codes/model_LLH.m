function [LLH] = model_llh(params, data, N, T)
p.rho1 = params(1);
p.rho2=params(2);
p.phi1 = params(3);
p.phi2=params(4);
p.beta=params(5);
p.sigma_eps = params(6);
p.sigma_A = params(7);
p.sigma_B = params(8);

T = min(T, length(data));
data_logA=log(data(:,1));
data_B=data(:,2);

rng(0)
lr_sim=5000;
x_distn=zeros(lr_sim+3,1);
distn_shocks=p.sigma_eps*randn(lr_sim+3,1);
for t=3:lr_sim+3
    x_distn(t)=p.rho1*x_distn(t-1)+p.rho2*x_distn(t-2)+p.phi1*distn_shocks(t-1)+...
        p.phi2*distn_shocks(t-2);
end

particles=zeros(T,N,6);
llhs=zeros(T,1);
init_sample=randsample(lr_sim,N);

%initial states:
particles(1,:,1)=x_distn(init_sample+2);
particles(1,:,2)=x_distn(init_sample+1);
particles(1,:,3)=x_distn(init_sample);
particles(1,:,4)=distn_shocks(init_sample+2);
particles(1,:,5)=distn_shocks(init_sample+1);
particles(1,:,6)=distn_shocks(init_sample);

llh=normpdf(data_logA(1),particles(1,:,1),p.sigma_A).*...
    normpdf(data_B(1),p.beta*particles(1,:,1).^2,p.sigma_B);
llhs(1)=log(sum(llh))-log(N);


for t = 2:T
    %%% Prediction:
    shocks = p.sigma_eps*randn(1,N);
    particles(t,:,1)=p.rho1*particles(t-1,:,1)+p.rho2*particles(t-1,:,2)+...
        p.phi1*particles(t-1,:,4)+p.phi2*particles(t-1,:,5)+...
        shocks;
    particles(t,:,2)=particles(t-1,:,1);
    particles(t,:,3)=particles(t-1,:,2);
    particles(t,:,4)=shocks;
    particles(t,:,5)=particles(t-1,:,4);
    particles(t,:,6)=particles(t-1,:,5);
    
    %%% Filtering:
    llh = normpdf(data_logA(t),particles(t,:,1),p.sigma_A).*...
        normpdf(data_B(t),p.beta*particles(t,:,1).^2,p.sigma_B);
    weights = exp(log(llh)-log(sum(llh)));
    if sum(llh)==0
        weights(:)=1/legth(weights);
    end
    % store the log(mean likelihood)
    llhs(t) = log(sum(llh))-log(N);
    
    %%% Sampling:
    samples=randsample(N,N,true,weights);
    particles(t,:,:) = particles(t,samples,:);
 
    
end

LLH = sum(llhs);