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

2020-05-30 21:40

Added better image -> seq and now testing image -> seq and seq blocks.
Tried a bunch of runs with O2. BAD divergence behavior. DO NOT BE TEMPTED.
O2 will not work with current model (O1 seems fine).

2020-05-31 10:53

I think something is bugged with test loss reporting. Test loss is order
of magnitude lower consistently at start of training.
improve_seq looks better (based on test samples) but has higher test loss.
This may be related to test loss reporting issue.
Might be related to distributed loader or fp16.
Also, fix loss reporting in general and in distributed case.

2020-05-31 15:11

All runs in 5-30 have a bug which effects test/train error (failed to randomly
partition samples correctly). Retrying now.

2020-06-01 00:58

Ran many more tests. ImageToSeq and SeqToImage both seem to work quite well
(after adjusting ImageToSeq to have an initialization which initially
ignores the image - this helps with avoiding loss of information).
If sequence blocks are added (transformers), then test loss is SUPER jumpy.
However, in general, sequence blocks seem to work (after things cool a bit).
This looks like BN effects, but it isn't clear to me why sequence blocks
specifically would cause this. Right now, we use separate channels for
SeqToImage and BN is directly applied to this. Seems related but hard to see
how.

I am now (a little) concerned that these results might just come down to faster
learning (as opposed to better converged accuracy). We aren't currently fully
converging and there is a difference in the level of convergence. I think way
more epochs are required and that cos annealing is a better idea.  We need more
principled approach to LR.

To try/TODO:
 - Make background black and switch to RELU output activation.
 - Eliminate input transformer
 - Switch to standard descending channel type architecture.
 - Add SeqToImage to all channels with an overall weight much like how
   ImageToSeq works (currently small number of channels are concatenated).
 - Correct frequencies of ImageToSeq and SeqToImage?
 - Correct (base) seq size?
 - Correct (base) ch count?
 - Correct (base) depth?
 - Correct expand ratio?
 - Correct initial dim?
 - Is staggering/interlacing actually improving efficiency?
 - How should the model be scaled up (ch multiplier, depth multiplier, seq
   size multiplier)?
 - I think lr schedule isn't ideal. We aren't converging right now. I think this
   is because the decay phase needs to be at a higher learning rate.*** 
Carried over from before:
 - better strategies for attention
 - "intersection" modules/activation


Implemented better system for lr (I think). Now running with MANY more epochs
(200).

Runs (2020-05-31):

standard_run: no SeqToImage or seq blocks

just_image_to_seq: just SeqToImage

new_improve_seq: SeqToImage and seq blocks, but with initialization change
(negative bias)

just_seq_blocks: just seq blocks

improve_seq_half_lr: same as new_improve_seq but with half lr (95% sure, but
this might actually just be an old run)

2020-06-01 11:23

Looks like big runs will converge to a satisfactory degree - this
is probably near the correct full convergence learning schedule.
I think 100 epoch training should be sufficient for experiments.
ImageToSeq is definitely very effective. Outputs now look pretty good -
specular highlights are showing up and spheres look pretty crisp.
Background not as good as I would expect.

To try/TODO:
  - List from before (yesterday)
  - Dropout (on seq blocks) and other regularization
  - Different seq blocks/more residual blocks somehow

Runs (2020-06-1 part_0):

just-image-to-seq_big: big run with ImageToSeq and SeqToImage
orig_big: big run without ImageToSeq
(has bug mentioned below)
I deleted the tensorboard files for these runs prior to copying them locally
(oops). TLDR: ImageToSeq ended at 6e-4 and without ended up at 1.2e-3.
Both converged train to about 1e-4.

2020-06-01 12:53

There was a bug in ImageToSeq involving the pooling. This was fixed and will be
tested (independently of other changes). We previously incorrectly used the
same overall_weight for all sequences in ImageToSeq.

2020-06-01 17:49

Runs (2020-06-1 part_1):

add_seq_to_image: Added SeqToImage for each channel using "is_cross_attn" -
based on approach of ImageToSeq.

big_new_run: Same as just-image-to-seq_big, but 100 epochs only and bug fixed.
(new baseline).

add_seq_to_image_quarter_lr: add_seq_to_image diverged, so I changed learning
rates to be 1/4.

add_seq_to_image_fixed: Apparently there was a bug where the value for cross
attention was added in twice. This run has it fixed. Apparently that
caused the divergence (which is a bit interesting, in theory,
weights should have been able to adapt pretty well to this situation).


Runs (2020-06-1 part_2):

only_descending_ch: run with only_descending_ch (half at each step rather
than increasing initially) and 1024 starting channels  with 256 starting
attn ch (initial-attn-ch set to 512). I had to lower batch size and
this is a decent amount slower. 

no_base_transformer: removed base transformer to see if it is important.

2020-06-02 20:05

Interestingly, no_base_transformer eventually diverges. This happens somewhat
after peak lr and appears to be related to mixed precision training.  I tried
reducing lr slightly to prevent this (0.85 multiplier), but this still diverged
at almost the same point in training.  Other than eventual sudden divergence,
no_base_transformer was promising.  Then, I tried keeping the transformer, but
with only one layer. This performs better than no base transformer and hasn't
(yet) diverged. My current theory is that parameter sharing is a bust. For now,
I will run with one layer, later I will try multiple layers, but with different
parameters for each layer.  I think this may also extend to seq blocks, so I am
trying some runs with 1 layer seq blocks.

add_seq_to_image_fixed didn't diverged but performed poorly, so I tried
restricting the channels which attn is added to. This resulting in almost the
exact same result, making me suspect that the poor results are due to the attn
having no effect. I further guessed this was because the initialization of
the mix bias was too low. I changed the init from -10.0 (sigmoid -> 4.5e-05) to
0.0 (sigmoid -> 0.5). This works very well (add_seq_to_image_mix_bias),
and add_seq_to_image seems like the best approach. This run was done with
a subset of the channels having attention added to them, but I am guessing that 
adding attention to all channels is effective.

only_descending_ch performed somewhat better, but was a bigger model. Thus, it
isn't clear that this a better way to distribute computational resources.
There are roughly three possible strategies I can think of:
 - Steadily expand channels and then half channels every upsample
 - Keep channels constant and then half channels every upsample
 - Half channels every upsample from the start
Big GAN cuts channels in half from the start for 128x128 but for higher
resolution they add the additional block at the start with same channels as
the block after it. This is the keep channels constant strategy, but
with very few constant channels blocks. I have been using steady expand,
but have tested the last strategy. One advantage of the last two stategies is
that they keep computational load more consistant throughout the net.

Runs (2020-06-02):

add_seq_to_image_new: same as add_seq_to_image_fixed, but with attn on a subset
of channels.

add_seq_to_image_mix_bias: same as add_seq_to_image_new, but with a different
initialization for the mix bias.

no_base_transformer_reduced_lr: same as no_base_transformer, but with 0.85 lr
multiplier. Still diverges.

minimal_base_transformer: 1 layer base transformer.

single_layer_transformer_and_add_all: 1 layer base transformer, 1 layer
seq blocks (these have been disabled for most of previous runs),
add_seq_to_image on all channels (not on a subset like in add_seq_to_image_new).
Basically kitchen sink run.

Random divergence strikes again!!! Again at about the same batch. Symptoms are
continuous gradient overflow and then eventually loss scale going to zero.
Set min loss scale of 128 - hopefully this will solve the issue and previous
tests can be rerun...

Min loss scale may help in some cases, hard to tell. However, it definitely
doesn't fix divergence in all cases. I still can't train without the input
transformer (at least without lowering learning rates). For now I will just
accept that it must be used.

image_to_seq block is critical (removing block vastly increases test loss,
~5.7e-4 -> ~1.5e-3).

I still can't get no_base_transformer to converge. Very bizarre. I will
keep it for now. I also might try removing the base transformer and using fp32
training (to see if that is worth the trouble).

I think seq blocks should work but are overfitting. I might
reintroduce seq blocks, but with some form of strong regularization
(maybe dropout or a variant of stochastic depth).

Some data was lost from compute going offline. I think
single_layer_transformer_and_add_all performed pretty poorly. I also ran a
version which was the same thing but with a subset of channels for attn.
Both performed meh, indicating to me that current SOTA is
SeqToImage (on subset of channels), ImageToSeq, no seq blocks,
add attn (as opposed to concat), and channel structure isn't too important
(strictly decreasing vs increasing then decreasing).

One other thing I want to test is removing a linear block from SeqToImageStart
(called _feat_to_output).

Huh, seems like everything is diverging on the instance with 1080tis. Seems
suspect. I think there may be some sort of underlying issue. For now I
will kill that instance and rerun some of the experiments. I am putting these
experiments in the "questionable_divergence" directory.

The mix bias for ImageToSeq was too high in several previous experiments. I
some tuning requried. Also, I am thinking a longer warm up pct might be
better for training efficiency. TODO: get lr sched and mix bias exactly right.
Then tune with/without transformer, sequence blocks, count of attn ch,
overall ch count, sequence size etc...

Increasing seq size makes learning smoother, but otherwise doesn't seem to
have much effect. I have tuned the bias to -7.0 which seems pretty successful.
Some runs were executed with -5.0 and this can sometimes work (value is pretty
sensitive).  When no_base_transformer diverges, the core issue seems to be 
something with bn (train converges and test outputs all black). To hopefully
address this issue, we now reset the running stats at the start of each
epoch. I will rerun the no_base_transformer test (for the 40th time oof).
I am pretty happy with -7.0 mix bias and current lr schedule (assuming
things keep converging). Direct next steps:
 - retest adding seq blocks.
 - retest remove base transformer
 - save all run settings in tensorboard (all args + commit hash)

2020-06-05:

Interesting finding: multi gpu training is important. This is likely due
to sequence length issues. Training with 4 gpus makes things much more stable.
O1 doesn't appear to make things much faster (maybe slightly). I will do some
profiling including with O2 and see what we find. O2 doesn't yield good
results. I don't think O1 is much faster than O0, but I will test and find out.
This should inform correct opt level.
Larger seq size didn't seem to improve anything, but I might rerun with new
settings. I am currently running smaller seq size.

O0 is just as fast as O1 (smaller batch size outweighed by no need for
conversions I assume).

Interestingly, O0 and O2 both perform about the same and perform worse than
O1. Perhaps O2 is too much loss of accuracy and O1 also has a regularization
effect?

2020-06-06 13:46

To fix sequence length bn issues, masking should work. This has now been
implemented.

I ran a test reintroducing seq blocks. seq blocks had a moderate negative
effect. This may be related to overfitting. Mitigation via regularization
or sharing parameters may be effective. These effects might change with
the introduction of masking.

I have some ideas to alter the input block - will test.

Large seq size performs poorly (test divergence). No input transformer also
performs poorly (general overfitting).

2020-06-07 16:36

Nonlocal block performs poorly. Increasing seq frequency doesn't change
performance much (test perf similar, train slightly worse - less overfit but
very small difference).

2020-06-07 22:23

squeeze excitation lowers performance by a lot. Position channels lowers
performance substantially (this is interesting, I would expect the effect to be
smaller than it is and to be either helpful or no effect - probably related to
overfitting).

I have some ideas for a squeeze excitation like block using attention which
may work better. Basically, compute channel wise multipliers per sequence
element (also using GAP values perhaps) and then weight multipliers using
attention per pixel.

I have some ides for better seq blocks. Let input be x.
y = swish(contract_linear(swish(expand_linear(x))))
x = cross_attn(y, x)

This is pretty similar to transformer blocks but with a few important tweaks.
I think this will work better.

2020-06-08 09:50

Bizarrely, combining no_se and no_position_ch leads to poor performance.  I
think this may be related to the fact that I ran the combined run with 2 gpus
and the individual runs with 4 gpus. This isn't the case. Runs with 2 gpus
perform similarly to runs with 4 gpus (except for the lr schedule which varies
somewhat). There was divergence with 4 gpus, but after reducing learning rates,
the results with 4 and 2 gpus were similar. It looks like the learning rate
is right on the edge of too high.

For now, I will keep position_ch.

I adjusted channels so that it stays the same and then goes down.
This performed slightly worse actually. Not sure why. Switching to path
traced images.

2020-06-28 21:25

Rendered 300_000 images. Now running baselines. To get training to work with
long sequence lengths, progressive training is needed. Also, 0.25 lr-multiplier
to avoid divergence.

TODO:
 - even lower lr.
 - see if divergence etc issues go away when removing image-to-seq
 - see if divergence etc issues go away when removing seq-to-image
   (image-to-seq can also be removed in this case)
 - see if divergence etc issues go away when changing gain bias based on
   sequence len.
 
Fixed bug with attn code - could have caused issues with sequence length
variability: testing now
