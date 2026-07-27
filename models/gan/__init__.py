
from .enhance import CrackEnhanceGAN, create_enhance_gan
from .cyclegan import CrackCycleGAN, create_cyclegan
from .srnet import CrackSRNet, create_srnet

GAN_TYPES = ("enhance", "cyclegan", "super_resolution")
