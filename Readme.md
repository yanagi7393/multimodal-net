## Project1: Sound -> Image
```
1. Model Structure:
  - [x] use spatial dropout
  - [-] implement U-net++ structure　----------(PURGED)
  - [x] implement resnet block
  - [x] implement inverted residual block 
  - [x] implement self-attention
  - [x] implement squeeze exitation block
  - [ ] implement handling spectal normalization
  - [x] implement Multi method normalization
  - [ ] implement Batch-Instance normalization
  - [-] implement Cycle-GAN structure  ----------(NOT YET, NOW PURGED)
  - [ ] implement GAN structure
2. Preprocessing
  - [x] SOUND: implement STFT handling
  - [ ] IMAGE: implement image augumentation method 
3. DATASET
  - [x] video splitter
      - 1. extract frames, sound from video.
      - 2. parallelize
  - [x] sound and frame custom data loader for build datasets
  - [x] multiple audio files handle on stft
4. Training
  - [_] LATER
5. Testing
  - [_] LATER
6. APPLY MULTI GPUs
  - [_] LATER
7. make package
  - [ ] Define command use fire library
  - [ ] Define & Build docker image
```

## Project2: Sound <-> Image Cycle 

## Project3: Any(Sound or Image) -> Inner similar vector -> Any(Sound or Image)
