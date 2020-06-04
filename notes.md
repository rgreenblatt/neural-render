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
keep it for now. I think seq blocks should work but are overfitting. I might
reintroduce seq blocks, but with some form of strong regularization.
