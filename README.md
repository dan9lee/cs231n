# cs231n
Final Project

Project based primarily off code from original AttnGAN (https://github.com/taoxugit/AttnGAN)

max_sv and snconv2d are code directly from https://github.com/godisboy/SN-GAN. We used these to apply Spectral Norm in our implementation of model_SN_FINAL and model_SN_on_D1_FINAL

model_SN_FINAL, model_SN_on_D1_FINAL, trainer_WGAN_updated, model_SN_full_channels_dropout, and losses_WGAN are all modified code sourced from  original AttnGAN implementation

model_SN_FINAL applies spectral norm after all Conv2D operations in each of the three discriminator branches

model_SN_on_D1_FINAL applies spectral norm only after the Conv2D operations in the first discriminator branch

Unssuccessful models:
model_SN_full_channels_dropout - modified the AttnGAN architecture to add spectral normalization to discriminators, removes batchnorm in the discriminators, adds greater filter depth, applies some dropout in the  generator.

WGAN-GP: includes trainer_WGAN_updated and losses_WGAN 
trainer_WGAN_updated is the trainer used for WGAN (updated to include loss logging)
losses_WGAN uses the Earth-mover distance and gradient penalty from https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py

