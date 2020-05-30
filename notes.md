2020-05-29 16:04

Finished spheres_lower_lr_02, spheres_lower_lr_O1, spheres_O1, spheres_O2
(experiment where we reduce lr by factor of 0.25 and check O2 vs O1).
Placed under "spheres_initial"
Determined lr should be halved and O1 is best (possible that higher lr will
have nice regularization effect). Now testing with more data and observing
effect of tanh over softmax. Also testing effect of seq_to_image blocks.
NOTE: we use lr-multiplier of 0.85 on instance with lower batch size
(due to less vram).

2020-05-29 18:19

Oops, screwed up the lr schedule. I will rerun, results under "broken_lr".

2020-05-29 21:04

Values can get pretty high now, consider switching loss function to something
more "perceptual"
(maybe something like smooth L1 loss scaled by the value of the image + 1?)
Also, consider switching to exp(x) from celu(x) as output activation

2020-05-30 01:31

Using a more perceptual loss function worked well based on visual inspection
(see perceptual_loss run). Using an exp function at the output performed
very poorly (see use_exp). After starting up a big run using tanh, I
switched from tanh to not using tanh base on preliminary results
(see big_run_no_tanh and big_run).

To try/TODO:
 - better strategies for attention
 - "intersection" modules/activation
 - reintroduce image_to_seq 
 - reintroduce transformers on sequence throughout
 - "interlace" or occasionally use the previous two to avoid delays.
 - Tune lr better for new loss function 
 - MOAR data???? 
 - 4x4 is worse than it should be - better start block?
