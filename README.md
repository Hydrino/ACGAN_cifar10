# ACGAN_cifar10
A PyTorch implementation of Auxiliary Classifier GAN to generate CIFAR10 images. 

## Literature survey

- The orginal ACGAN paper - [https://arxiv.org/abs/1610.09585]
- Code heavily inspired by <a href = "https://github.com/pytorch/examples/blob/master/dcgan/main.py">dcgan pytorch</a> repo. 
- A lot of tuning and hacks borrowed from <a href = "https://github.com/soumith/ganhacks">this</a> awesome repo.

## Results

After training for 50 epochs with batch size 100, following are the results of some of the classes.

- horses
![images/horses_gen.png](images/horses_gen.png)

- cars
![images/gen_cars.png](images/gen_cars.png)

- Frogs
![images/frogs_gen.png](images/frogs_gen.png)

- trucks
![images/trucks_gen.png](images/trucks_gen.png)
