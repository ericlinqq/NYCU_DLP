# <center>Lab4 Conditional VAE for video prediction</center>  
### <center>311554046 林愉修</center>  
## **1. Introduction**  
  An VAE model for video prediction based on **SVG-FP** model ([**Emily Denton & Rob Fergus, 2018**][1]), which the model architecture is as the figure shown below:  

![model architecture](https://i.imgur.com/R6WRll3.png)  

>  This model take past frame $x_{t-1}$ as input for encoder, which will output an latent vector $h_{t-1}$, and we will sample $z_t$ from the posterior we learned. Eventually, we take $h_{t-1}$ and $z_t$ with action and position (the condition) as the input for decoder, which the output is expected to be the next frame $\hat{x_t}$.  
  
  We need to implement this VAE model. Furthermore, the **action and position data** are used as the condition to make this an CVAE model, **Teacher forcing** and **KL annealing schedules** are required in the training function.  
  Once the model finish training, we test it out by making gif image and prediction at each time step to see if the model prediction is close to the ground-truth image.  
  
>  Note that in the testing phase, we sample $z_t$ from fixed prior, which we assumed here is a **standard normal distribution**.  


## **2. Derivation of CVAE**  
  The derivation of the CVAE objective function:  
To determine $\theta$, we would intuitively hope to maximize $p(X|c;\theta)$  

$$ p(X|c;\theta) = \int p(X|Z,c;\theta) p(Z|c)dZ $$  

  This however becomes difficult since the integration over $Z$ would have no closed-form solution when $p(X|Z,c;\theta)$ is modeled by a neural network.  

  By the chain rule of probability, we know:  

$$ p(X,Z|c;\theta) = p(X|c;\theta) p(Z|X,c;\theta) $$   
$$\implies \log p(X,Z|c;\theta) = \log p(X|c;\theta) + \log p(Z|X,c;\theta) $$  
$$ \implies \log p(X|c;\theta) = \log p(X,Z|c;\theta) - \log p(Z|X,c;\theta) $$  

  We next introduce an arbitrary distribution $q(Z)$ on both side and integrate over $Z$:  

$$ \int q(Z)\log p(X|c;\theta)dZ = \int q(Z)\log p(X,Z|c;\theta)dZ - \int q(Z) \log p(Z|X,c;\theta)dZ $$  

$$ = \int q(Z)\log p(X,Z|c;\theta)dZ - \int q(Z) \log q(Z)dZ + \int q(Z) \log q(Z)dZ - \int q(Z) \log p(Z|X,c;\theta)dZ $$  

$$ = \scr L \mit(X,c,q,\theta) + KL(q(Z)||(Z|X,c;\theta)) $$  

  where  

$$ \scr L \mit(X,c,q,\theta) = \int q(Z)\log p(X,Z|c;\theta)dZ - \int q(Z) \log q(Z)dZ $$   

$$ KL(q(Z)||(Z|X,c;\theta)) = \int q(Z) \log q(Z)dZ - \int q(Z) \log p(Z|X,c;\theta)dZ $$  

<br/>

$$ \therefore \log p(X|c;\theta) = \scr L 
\mit(X,c,q,\theta) + KL(q(Z)||p(Z|X,c;\theta)) $$  

$$ \implies \scr L \mit(X,c,q,\theta) = \log p(X|c;\theta) - KL(q(Z)||p(Z|X,c;\theta)) $$  

  Now, instead of directly maximizing the intractable $p(X|c;\theta)$, we attempt to maximize $\scr L \mit(X,c,q,\theta)$  

$$ \because \scr L \mit(X,c,q,\theta) = \int q(Z)\log p(X,Z|c;\theta)dZ - \int q(Z) \log q(Z)dZ $$  

$$ = \Bbb E_{Z \sim q(Z)}[\log p(X,Z|c;\theta)] - \Bbb E_{Z \sim q(Z)}[\log q(z)] $$  

$$ = \Bbb E_{Z \sim q(Z)}[\log p(X|Z,c;\theta)] + \Bbb E_{Z \sim q(Z)}[\log p(Z|c)] - \Bbb E_{Z \sim q(Z)}[\log q(z)] $$  

$$ = \Bbb E_{Z \sim q(Z)}[\log p(X|Z,c;\theta)] - KL(q(Z)||p(Z|c)) $$  

$$ \therefore \scr L \mit(X,c,q,\theta) = \Bbb  E_{Z \sim q(Z)}[\log p(X|Z,c;\theta)] - KL(q(Z)||p(Z|c)) $$  

  Because the equality holds for any choice of $q(Z)$ , we introduce a distribution $q(Z|X;\phi)$ modeled by another neural network with parameter $\phi$.  

  To maximize  

$$ \scr L \mit(X,c,q,\theta, \phi) = \log p(X|c;\theta) - KL(q(Z|X;\phi)||p(Z|X,c;\theta)) $$  

  which amounts to maximizing  

$$ \Bbb  E_{Z \sim q(Z|X;\phi)}[\log p(X|Z,c;\theta)] - KL(q(Z|X;\phi)||p(Z|c))$$  


## **3. Implementation details**  
**1. Model implementation**  
* **Encoder**  
  The frame encoder uses the same architecture as **VGG16** ([**Simonyan & Zisserman, 2015**][2]) up until the fourth pooling layer and the final convulutional layer contain a **conv4-128** with no padding, besides, the activation function is replaced with $Tanh$, as the figure shown below.  
![VGG16](https://i.imgur.com/EhO27pi.jpg)  
```python=
# Final layer
self.c5 = nn.Sequential(
        nn.Conv2d(512, dim, 4, 1, 0),
        nn.BatchNorm2d(dim),
        nn.Tanh()
        )
# Maxpool
self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
```  

  The input $x$ shape here is **($b$, 3, 64, 64)**; as for the output, the latent vector $h$ shape is **($b$, 128, 1, 1)**.  
  
  > $b$ represents the batch size.
  
* **Decoder**  
  The decoder is basically a mirrored version of the encoder with pooling layer replaced with **spatial up-sampling** and a **sigmoid** output layer, as the figure shown below.  
![Decoder](https://i.imgur.com/0rC0vWt.png

  The input $g$ shape here is **($b$, 128, 1, 1)** , which is as same as the output shape of the encoder;  the output $\hat{x}$ shape is **($b$, 3, 64, 64)**, which is as same as the input shape of the encoder.  

* **Reparameterization trick**  
  Since the LSTM in the inference model is expected to output the mean and variance of a Gaussian distribution, where $z$ is sampled from. This however is *non-differientiable*, that is, the model could not be trained **end-to-end**.  
  In order to solve this problem, we need to implement the so-called *"reparameterization trick"*, which simply drawn a $\epsilon$ from standard normal distribution, and the operation is as follow:
  $$ \epsilon \sim N(0, 1) $$  
  
  $$ z = \epsilon \ast \sigma + \mu$$  
  
  so that  
  
  $$ z \sim N(\mu, \sigma) $$  
  
  where $\mu$ and $\sigma$ are the output of the LSTM in the inference model.  
```python=
def reparameterize(self, mu, logvar):
    logvar = logvar.mul(0.5).exp_()
    eps = logvar.detach().new(logvar.size()).normal_()
    return eps.mul(logvar).add_(mu)
```  

* **Dataloader**  
  The dataloader loads a batch of the image sequence data (.png) and the condition data (.csv) every time.  
  For the image, we transforms it into a **tensor**, which would turn 8-bit lightness (0~255) into a **floating point (0~1)**. By concatenating all the image in every time step, we get a image sequence with the shape of **($t$, 3, 64, 64)**.  
  
  > $t$ represents the total time step for a image sequence.  

```python=
default_transform = transforms.Compose([
    transforms.ToTensor(),
    ])
```

  The condition data is composed of action (dimensionality 4) and position data (dimentionality 3). By concatenating them in every time step, we get a condition with the shape of **($t$, 7)**.  
  Since the encoder input shape is **($b$, 3, 64, 64)**, we need to **transpose** the image sequence between dimension 0 (time step) and dimension 1 (batch), so that the shape would be **($t$, $b$, 3, 64, 64)**. The condition data should also do the same thing, because we then directly concatenate it with the output $h$ from the encoder and $z$ from inference model, which is soon to be sended into the LSTM in prediction model.  
```python=
seq, cond = seq.transpose_(0, 1).to(device), cond.transpose_(0, 1).to(device)
```  

 
* **Condition**  
  As mentioned above, I simply **concatenates** the condition data with $h$ and $z$, as the input for the LSTM in prediction model.  
```python=
h_pred = modules['frame_predictor'](torch.cat([cond[i], h, z_t], 1))
```

* **KL annealing schedules**  
  Due to the sequential nature of text and video, an auto regressive decoder is typically employed in the
VAE. This is often implemented with a RNN(LSTM). This introduces one notorious issue when a VAE is trained using traditional methods: the decoder ignores the latent variable, yielding what is termed the ***KL vanishing problem***.  
  [Fu, Li & Liu et al. (2019)][3] hypothesize that the problem is related to the low quality of $z$ at the beginning phase of decoder training. A lower quality $z$ introduces more difficulties in reconstructing $x$ via Path A. As a result, the model is forced to learn an easier solution to decoding: generating $x$ via Path B only, as the figure shown below:  
![auto-regressive decoder](https://i.imgur.com/XxOWQvL.png)  
  It is natural to extend the negative of the objective function in VAE by intoducing a hyperparameter $\beta$ to control the strength of regularization:
$$ - \Bbb  E_{Z \sim q(Z|X;\phi)}[\log p(X|Z,c;\theta)] + \beta KL(q(Z|X;\phi)||p(Z|c)) $$  
three different schedules for $\beta$ have been commonly used for VAE.  

  * Constant schedule  
    The standard approach is to keep $\beta = 1$ fixed during training procedure, as it correspondes to optimizing the true VAE objective.
    
  * Monotonic annealing schedules  
    $\beta = 0$ is set at the beginning of training, and gradually increases $\beta$ until $\beta = 1$ is reached.
    
  * Cyclical annealing schedule  
    Split the training process into $M$ cycles, each starting with $\beta = 0$ and ending with $\beta = 1$.
  
```python=
class kl_annealing():
    def __init__(self, args):
        if not args.kl_anneal_cyclical:
            args.kl_anneal_cycle = 1 
        period = args.niter / args.kl_anneal_cycle
        step = 1. / (period * args.kl_anneal_ratio)
        self.L = np.ones(args.niter)
        self.idx = 0
        
        for c in range(args.kl_anneal_cycle):
            v, i = 0.0, 0
            while v <= 1.0 and (int(i+c*period) < args.niter):
                self.L[int(i+c*period)] = v
                v += step
                i += 1

    def update(self):
        if self.idx < len(self.L)-1:
            self.idx += 1

    def get_beta(self):
        beta = self.L[self.idx]
        return beta
```  
This class would construct an instance with variable which is a list contains the kl annealing weight in every epoch, and update function updates the index variable, which the get_beta() function uses as the index to access the beta in the list.  

**2. Teacher forcing**  
  In the early phase of the training process, The model capability of prediction is low, if one unit output an bad result, it would definitely affect the learning of the following units.  
  Teacher forcing is a method for efficiently training RNN models, which uses the ground-truth as input, instead of model output from a prior time step as input.
  By using teacher forcing, the model would converge in the earlier iteration, moreover, the training is more stable than using free-running.  
  
  Since this method highly depends on the ground-truth label, it performs better during the training process. But in the testing phase, without the support of the ground-truth, terrible results may probably be seen.

## **4. Results and discussion**  
**1. Results of video prediction**  
**(a) Make videos or gif images for test result** 

**(b) Output the prediction at each time step**  

**2. Plot the KL loss and PSNR curves during training**  

**3. Discuss the results according to your setting of teacher forcing ratio, KL weight,
and learning rate.**  



[1]: https://arxiv.org/abs/1802.07687 "arxiv"
[2]: https://arxiv.org/abs/1409.1556 "arxiv"
[3]: https://aclanthology.org/N19-1021.pdf "aclanthology"