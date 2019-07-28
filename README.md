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
Create folder 'data' and download celebA dataset.

Run train-gd.py train-e.py to train the generator+discriminator and encoder sequentially.

```python
python3 train-gd.py
python3 train-e.py
```

Modify and run test.py for evaluation.

## Results

<a href="https://join.slack.com/t/tensorlayer/shared_invite/enQtMjUyMjczMzU2Njg4LWI0MWU0MDFkOWY2YjQ4YjVhMzI5M2VlZmE4YTNhNGY1NjZhMzUwMmQ2MTc0YWRjMjQzMjdjMTg2MWQ2ZWJhYzc">
<div align="center">
	<img src="results/xxx" width="80%" height="50%"/>
</div>
</a>


