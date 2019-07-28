# dcgan_with_encoder
Implementation of DC-GAN with Encoder using TensorLayer 2.0

## Requirement
- Python3
- TensorFlow==2.0.0a0
- TensorLayer==2.1.0
- CPU or NVIDIA GPU + CUDA CuDNN

## Train the Model

Clone this repo

```python
git clone https://github.com/YHQpkueecs/dcgan_with_encoder.git
```
Create a folder `data` and download celebA dataset.

Run `train-gd.py` `train-e.py` to train the generator+discriminator and encoder sequentially.

```python
python3 train-gd.py
python3 train-e.py
```

Modify and run `test.py` for evaluation.

## Results
Randomly generate 64 faces
![](https://github.com/YHQpkueecs/dcgan_with_encoder/blob/master/image/train_20.png)

Interpolate between two randomly generated faces
![](https://github.com/YHQpkueecs/dcgan_with_encoder/blob/master/image/vis_1.png)
![](https://github.com/YHQpkueecs/dcgan_with_encoder/blob/master/image/vis_2.png)
![](https://github.com/YHQpkueecs/dcgan_with_encoder/blob/master/image/vis_3.png)

Encoding and interpolation for two selected images 
![](https://github.com/YHQpkueecs/dcgan_with_encoder/blob/master/image/vis.png)
