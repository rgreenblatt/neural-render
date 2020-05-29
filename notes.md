2020-05-29 16:04

Finished spheres_lower_lr_02, spheres_lower_lr_O1, spheres_O1, spheres_O2
(experiment where we reduce lr by factor of 0.25 and check O2 vs O1).
Determined lr should be halved and O1 is best (possible that higher lr will
have nice regularization effect). Now testing with more data and observing
effect of tanh over softmax. Also testing effect of seq_to_image blocks.
