# Cat/dog image classification
Kaggle competition (late submission) trying various CNN structures. 
Both setting up different NN architectures myself as well as using transfer learning to make use of
pretrained models on the imagenet dataset. <br>

# setup
I have used Google Colab to get access to free GPU and use Keras with Tensorflow backend to train models.
Google Colab provides data science VM's with a lot of packages pre-installed. 
I've added a requirements file containing the contents of !pip freeze to make it reproducable in the future.
Some modules are installed from the notebooks itself.

# Different techniques tried
- Loading and preprocessing vs image data generation on the fly.
- Comparing speed GPU/TPU
- Cyclical learning rates vs learning rate decay vs standard Adam (with decay hyper tuning)
- Effect of nr of observations on test performance
- Image augmentation to prevent overfitting
- Image sizes and effect on train duration as wel ass test performance
- Early stopping to prevent overfitting
- Tensorboard to manage training
- Creating own residual blocks vs using keras resnet application
- Training model from scratch vs using transfer learning using pretrained model on imagenet dataset
- Trying different learning rates by freezing and unfreezing layers trainability
- Ensambling different models
- Clipping output (to prevent high penalty of log loss measure)

# Results accuracy and log loss on hold out set (5k images)
acc: 0.9252, loss: 0.2158 simplified VGG-model using data augmentation and 100x100 images <br>
acc: 0.9416, loss: 0.1520 cyclical learning rate triangular 0.00001-0.001 <br>
acc: 0.9676, loss: 0.0977 cyclical lr optimized triangular 0.00003-0.00035 <br>
acc: 0.9753, loss: 0.0819 increasing pixels to 224x224 <br>
acc: 0.9688, loss: 0.0829 resnet50 own implementation <br>
acc: 0.9868, loss: 0.0484 resnet50 transfer learning imagenet <br>
acc: 0.9929, loss: 0.0411 resnet 50 transfer learning different learning rates per layer <br>

The final submission got my to the top ~12%, although in an offline competition from a couple 
of years ago so that doesn't really count.

# TODO's
If this were to be a live competition, there would be more I would attempt. 
For now the time investments does not weigh up to the additional learning 
that would come from these attempts. <br>
Although it is of course interesting to see how well you could make a model perform.

- Try inception nets and its variants
- Try transfer learning from VGG and newer models
- Collect cat/dog pics from different datasets (e.g. imagenet) and see how the enriched dataset performs
- Ensamble more models together, try some voting techniques
- Do more hyper param tuning
- Do better inspection on wrongly classified (e.g. filter out non cat/dog images that are there)
- Try object detection first then fit model on the fitted objects
- Try cycling lr rates when systematically unfreezing layers of pretrained weights

