# <center>Lab4 Conditional VAE for video prediction</center>  
### <center>311554046 林愉修</center>  
## **1. Introduction**  
  An VAE model for video prediction based on the research of [**Emily Denton and Rob Fergus (2018)**][1], which the model architecture is as the figure shown below:  

![model architecture](https://i.imgur.com/R6WRll3.png)  

> This model take past frame $x_{t-1}$ as input for encoder, which will output an latent vector $h_{t-1}$, and we will sample $z_t$ from the posterior we learned. Eventually, we take $h_{t-1}$ and $z_t$ with action and position (the condition) as the input for decoder, which the output is expected to be the next frame $\hat{x_t}$.  
  
  We need to implement this VAE model. Furthermore, the **action and position data** are used as the condition to make this an CVAE model, **Teacher forcing** and **KL annealing schedules** are required in the training function.  
  Once the model finish training, we test it out by making gif image and prediction at each time step to see if the model prediction is close to the ground-truth image.  
  
> Note that in the testing phase, we sample $z_t$ from fixed prior, which we assumed here is a **standard normal distribution**.  


## 2. **Derivation of CVAE**  
The derivation of the CVAE objective function:  
To determine $\theta$, we would intuitively hop to maximize $p(X|c;\theta)$  

$$ p(X|c;\theta) = \int p(X|Z,c;\theta) p(Z|c)dZ $$

This however becomes difficult since the integration over $Z$ would have no closed-form solution when $p(X|Z,c;\theta)$ is modeled by a neural network.  

By the chain rule of probability, we know:  

$$ p(X,Z|c;\theta) = p(X|c;\theta) p(Z|X,c;\theta) $$ 
$$\implies \log p(X,Z|c;\theta) = \log p(X|c;\theta) + \log p(Z|X,c;\theta) $$
$$ \implies \log p(X|c;\theta) = \log p(X,Z|c;\theta) - \log p(Z|X,c;\theta) $$

We next intrduce an arbitrary distribution $q(Z)$ on both side and integrate over $Z$:  

$$ \int q(Z)\log p(X|c;\theta)dZ = \int q(Z)\log p(X,Z|c;\theta)dZ - \int q(Z) \log p(Z|X,c;\theta)dZ $$
$$ = \int q(Z)\log p(X,Z|c;\theta)dZ - \int q(Z) \log q(Z)dZ + \int q(Z) \log q(Z)dZ - \int q(Z) \log p(Z|X,c;\theta)dZ $$
$$ = \scr L \mit(X,c,q,\theta) + KL(q(Z)||(Z|X,c;\theta)) $$

where  

$$ \scr L \mit(X,c,q,\theta) = \int q(Z)\log p(X,Z|c;\theta)dZ - \int q(Z) \log q(Z)dZ $$
$$ KL(q(Z)||(Z|X,c;\theta)) = \int q(Z) \log q(Z)dZ - \int q(Z) \log p(Z|X,c;\theta)dZ $$
$$ \therefore \log p(X|c;\theta) = \scr L 
\mit(X,c,q,\theta) + KL(q(Z)||p(Z|X,c;\theta)) $$
$$ \implies \scr L \mit(X,c,q,\theta) = \log p(X|c;\theta) - KL(q(Z)||p(Z|X,c;\theta)) $$

Now, instead of directly maximizing the intractable $p(X|c;\theta)$, we attempt to maximize $\scr L \mit(X,c,q,\theta)$  

$$ \because \scr L \mit(X,c,q,\theta) = \int q(Z)\log p(X,Z|c;\theta)dZ - \int q(Z) \log q(Z)dZ $$
$$ = \Bbb E_{Z \sim q(Z)}[\log p(X,Z|c;\theta)] - \Bbb E_{Z \sim q(Z)}[\log q(z)] $$
$$ = \Bbb E_{Z \sim q(Z)}[\log p(X|Z,c;\theta)] + \Bbb E_{Z \sim q(Z)}[\log p(Z|c)] - \Bbb E_{Z \sim q(Z)}[\log q(z)] $$
$$ = \Bbb E_{Z \sim q(Z)}[\log p(X|Z,c;\theta)] - KL(q(Z)||p(Z|c)) $$
$$ \therefore \scr L \mit(X,c,q,\theta) = \Bbb  E_{Z \sim q(Z)}[\log p(X|Z,c;\theta)] - KL(q(Z)||p(Z|c)) $$

Because the equality holds for any choice of $q(Z)$ , we introduce a distribution $q(Z|X;\theta^{\prime})$ modeled by another neural network with parameter $\theta^{\prime}$.  
To maximize  
$$ \scr L \mit(X,c,q,\theta) = \log p(X|c;\theta) - KL(q(Z|X;\theta^{\prime})||p(Z|X,c;\theta)) $$
which amounts to maximizing  
$$ \Bbb  E_{Z \sim q(Z)}[\log p(X|Z,c;\theta)] - KL(q(Z|X;\theta^{\prime})||p(Z|c))$$


## 3. **Implementation details** 
**1. Model implementation**  
* Encoder  

* Decoder  

* Reparameterization trick  

* Dataloader  

* Condition  

**2. Teacher forcing**  


## 4. **Results and discussion**  
**1. Results of video prediction**  
(a) Make videos or gif images for test result  

(b) Output the prediction at each time step  

**2. Plot the KL loss and PSNR curves during training**  

**3. Discuss the results according to your setting of teacher forcing ratio, KL weight,
and learning rate.**  



[1]: https://arxiv.org/abs/1802.07687 "arxiv"