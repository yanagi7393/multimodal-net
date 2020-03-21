## Project1: Sound -> Image
```
1. Model Structure:
  - [x] use spatial dropout
  - [-] implement U-net++ structure　----------(PURGED)
   --> [x] implement partial skip connect between sound and image
  - [x] implement resnet block
  - [x] implement inverted residual block
  - [x] implement self-attention
  - [x] implement squeeze exitation block
  - [x] implement handling spectal normalization
  - [x] implement Multi method normalization
  - [x] implement Batch-Instance normalization
  - [x] implement spectral normalization
  - [-] implement Cycle-GAN structure  ----------(NOT YET, NOW PURGED)
  - [x] implement GAN structure (main model)
  - [x] implement "weights_init" to init VARIABLES
  - [x] implement WGAN-GP loss
2. Preprocessing
  - [x] SOUND:
    - [x] STFT handling
    - [x] Normalizer
  - [x] IMAGE: implement image Normalizer
3. DATASET
  - [x] video splitter
      - 1. extract frames, sound from video.
      - 2. parallelize
  - [x] sound and frame custom data loader for build datasets
  - [x] multiple audio files handle on stft
4. Training
  - [x] implement trainer
  - [x] implement save & load handler
5. Testing
  - [ ] LATER
6. APPLY MULTI GPUs
  - [ ] LATER
7. make package
  - [ ] Define command use fire library
  - [x] Define & Build docker image
```

## Project2: Sound <-> Image Cycle 

## Project3: Any(Sound or Image) -> Inner similar vector -> Any(Sound or Image)
