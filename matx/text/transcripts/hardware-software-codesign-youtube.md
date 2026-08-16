[00:00:00] I'm I'm here with Clive and Reiner. You

[00:00:01] two have both come over to my house

[00:00:02] often to talk about chips and hardware

[00:00:04] software co-design, but for some reason

[00:00:06] James Hill has decided to stick a camera

[00:00:08] in our face this time.

[00:00:09] Uh

[00:00:10] yeah, introduce yourselves.

[00:00:12] >> Yeah, I I I'm Reiner. I um I'm CEO of

[00:00:15] Matics. Uh we we make chips for

[00:00:17] uh large language models. Um and then

[00:00:19] before that I was doing um

[00:00:21] inference stack optimization at Google

[00:00:23] for for all of this.

[00:00:25] >> It's great to be here. Uh I'm a guy on X

[00:00:27] that likes to talk about hardware and

[00:00:29] software co-design.

[00:00:30] >> Yes, I think I think like one of the

[00:00:32] things you two have taught me a lot

[00:00:33] about is just this this whole hardware

[00:00:35] software design code space, right? So,

[00:00:37] Reiner, you wrote the Palm paper, um and

[00:00:39] I thought that was the best paper of

[00:00:40] like '23. Uh maybe '22, I don't know

[00:00:42] when it came out, but

[00:00:43] >> We had to stop publishing after that.

[00:00:44] >> Yeah, that was the last paper that

[00:00:46] Google published that was like valuable

[00:00:48] in infra.

[00:00:49] Um

[00:00:51] Yeah, and I thought that was super

[00:00:52] interesting. And then and then Clive,

[00:00:53] you you you know, you worked on a number

[00:00:55] of really interesting things.

[00:00:57] Um and you're co-designing for

[00:00:58] completely different space before.

[00:01:00] >> Yeah.

[00:01:01] >> Um what is what does co-design mean?

[00:01:03] >> So, co-design is when we can

[00:01:06] uh not only make the software better

[00:01:10] or the hardware better in isolation

[00:01:12] given

[00:01:13] just like

[00:01:14] a fixed workload or a fixed piece of

[00:01:16] hardware.

[00:01:17] Uh we want to make both better, and so

[00:01:19] we might want to

[00:01:21] uh change the hardware in a way that

[00:01:24] make some sacrifices today that will

[00:01:26] reap rewards in the future once we

[00:01:28] co-design for

[00:01:30] that future hardware.

[00:01:31] >> The thing that always confuses me is

[00:01:33] like uh how

[00:01:35] when you're doing co-design um

[00:01:38] what exactly are you doing, right? Like,

[00:01:39] you know, like it's like oh, are you

[00:01:40] just changing the shapes of the map

[00:01:42] moles, or like are you like what what

[00:01:44] what what kind of optimizations even

[00:01:46] exist in this space?

[00:01:47] >> I mean, I think like

[00:01:49] like the first thing is like the

[00:01:50] sacrifices is the key point, right?

[00:01:52] Like, uh what is a typical ML research

[00:01:55] paper? It's

[00:01:56] uh I've made some kind of my model

[00:01:57] architecture um and therefore all the

[00:01:59] metrics look better as a result. Like

[00:02:00] that's the archetypical thing.

[00:02:02] You can do that without code design.

[00:02:04] That is just a model research. It

[00:02:05] becomes code design when you say

[00:02:08] actually some of these metrics are

[00:02:09] getting worse,

[00:02:10] but that's because these are metrics run

[00:02:12] on the current generation of video GP.

[00:02:14] And so

[00:02:17] maybe I should look at a slightly more

[00:02:18] generalized metric like what is the

[00:02:21] the gate count of this of this

[00:02:23] architecture or what is the energy cost

[00:02:25] of this architecture or what's the

[00:02:26] intelligence per picojoule

[00:02:28] of this thing.

[00:02:29] Um

[00:02:31] And then so if you look at the more

[00:02:32] generalized

[00:02:34] metric, I think that sort of starts to

[00:02:35] approach giving you the flexibility to

[00:02:37] make all of the changes across the stack

[00:02:38] rather than just localized to the

[00:02:40] current generation of GPUs.

[00:02:41] So I think like very simple things are

[00:02:46] to take a case study. Like if you look

[00:02:48] at the difference between a Swish

[00:02:49] activation function,

[00:02:50] evaluating Swish requires

[00:02:54] evaluating some exponential functions.

[00:02:55] Those require

[00:02:57] function lookup tables, polynomial

[00:02:59] approximations. There's a lot of

[00:03:00] multipliers in them. Can be quite energy

[00:03:02] intensive. At the other extreme is to

[00:03:04] say ReLU is kind of the simplest

[00:03:06] activation function at all.

[00:03:08] Compared to greater than zero is

[00:03:10] extremely simple to evaluate.

[00:03:13] And so maybe it's cheaper in hardware to

[00:03:15] do to use ReLU rather than Swish,

[00:03:18] but maybe it's worse in in model

[00:03:19] quality. And then now your question is

[00:03:22] which one should I use? I want to do

[00:03:23] that on the basis of mega

[00:03:26] energy cost rather than rather than what

[00:03:29] on systolic array on GP.

[00:03:30] >> The example I love to give is

[00:03:33] you can always make your systolic array

[00:03:35] or tensor core wider in order to get

[00:03:38] more arithmetic into one place.

[00:03:41] But a lot of models are just not that

[00:03:43] wide.

[00:03:44] So

[00:03:46] if you increase the the size of the

[00:03:47] tensor core, you're not going to get any

[00:03:49] faster execution of that model. But if

[00:03:51] you

[00:03:52] tell them, all researchers,

[00:03:55] make the model wider. I know it's like

[00:03:56] not completely efficient for ML, but

[00:03:59] in net, once you also account for the

[00:04:01] hardware efficiency gains,

[00:04:03] it will be better.

[00:04:04] >> Why don't masters do that? Like one

[00:04:05] layer, uh 10 million wide or something

[00:04:07] like that.

[00:04:07] >> Well, if if you can build a 10 million

[00:04:09] wide uh tensor core that can

[00:04:13] run at the same power as a

[00:04:16] uh

[00:04:17] 128-wide tensor core, then obviously

[00:04:20] that's that's the right choice to make,

[00:04:22] but

[00:04:24] obviously that's not possible.

[00:04:25] >> So so a bit of a philosophical question,

[00:04:27] right? You see papers like um let's say

[00:04:29] Deep Seek V3, their attention mechanism

[00:04:31] was the first one at least of a public

[00:04:33] paper where the arithmetic intensity was

[00:04:35] equivalent to that of Hopper, right? In

[00:04:37] terms of how many memory how much memory

[00:04:38] bandwidth versus compute for the

[00:04:40] attention mechanism. Um you look at

[00:04:42] certain papers from other Chinese labs

[00:04:44] where they're like

[00:04:45] uh I think I think like Jen on Jen, I

[00:04:47] don't remember which lab it was. It

[00:04:48] might have been Alibaba, it might have

[00:04:49] been someone else, but Jen on Jen, their

[00:04:50] architecture was like very similar,

[00:04:52] except they went from like 100 layers to

[00:04:53] like 70 layers, and that just made

[00:04:55] inference like 20% faster, right? Um are

[00:04:58] these co-design because they're not

[00:04:59] really influencing the hardware? They're

[00:05:01] kind of looking at the hardware and the

[00:05:02] confines that it has, and and then

[00:05:04] making their model fit that hardware

[00:05:06] really well.

[00:05:07] >> Um I think a lot of people call a lot of

[00:05:09] things co-design, uh but in

[00:05:12] >> No true Scotsman is this that I be?

[00:05:15] >> Yeah. I think in our in both of our

[00:05:17] narrow views, uh co-design means that um

[00:05:21] instead of designing your model

[00:05:23] specifically for the hardware

[00:05:24] capabilities, like Deep Seek has done

[00:05:26] very well,

[00:05:28] uh or designing your hardware for

[00:05:32] uh models, which Nvidia's done very

[00:05:34] well, you're looking at both of the same

[00:05:36] time, and you're trading things off

[00:05:38] where you might actually make a

[00:05:39] sacrifice on model quality in order in

[00:05:42] order to get much more efficient

[00:05:43] hardware utilization, or the other way

[00:05:45] around, uh such that end-to-end you're

[00:05:48] doing better.

[00:05:49] >> It it did It seems to be and maybe just

[00:05:51] from the definition of it, uh you you

[00:05:53] get hardware companies that do

[00:05:54] co-design. Um

[00:05:56] pure model companies maybe less so or,

[00:05:59] you know, what's your read on that?

[00:06:00] >> I mean,

[00:06:01] you know, a public example is Trevor's I

[00:06:03] posted a tweet uh screenshotting an

[00:06:05] image of an email and that email was

[00:06:07] like, "Hey, here's what we want for

[00:06:09] hardware next to Nvidia." Yeah. Um and

[00:06:12] Nvidia did implement many of these

[00:06:13] things.

[00:06:15] Um I don't know what amount, but they

[00:06:16] did implement some of these things and

[00:06:18] then they released GPT-5.3 codex. Um X

[00:06:22] high is there are there more are there

[00:06:23] more words after Is it just codex?

[00:06:25] >> [laughter]

[00:06:26] >> All the numbers.

[00:06:28] >> Um anyways, you know, they released this

[00:06:29] model and they say it's co-designed with

[00:06:31] Blackbird. Is that Is that an example of

[00:06:32] a model lab

[00:06:34] with someone else's external hardware?

[00:06:35] >> Yeah. Seems like it's a good example.

[00:06:37] Um I think I think you do often kind of

[00:06:40] get these ping-pong relationships where

[00:06:42] you get feedback from from the model

[00:06:44] designers and then the hardware

[00:06:45] designers implement it. And then the

[00:06:46] hardware comes back and then the model

[00:06:48] designers get to take a look at it and

[00:06:49] then feedback kind of ping-pongs back

[00:06:51] and forth.

[00:06:52] Uh but uh maybe the larger ambition

[00:06:55] there is that we can look at both at the

[00:06:57] same time.

[00:06:57] >> How How do you How do you do that when

[00:06:59] the ML researchers don't even know what

[00:07:00] they're going to work on in like 6 to 9

[00:07:01] months and you're like hardware time

[00:07:05] cycles like I just The one I just

[00:07:06] mentioned is 3 years.

[00:07:07] >> Yeah, I mean, I would say like

[00:07:10] hardware designers kind of have to have

[00:07:13] to predict where the ML researchers are

[00:07:14] going and like maybe take a um

[00:07:17] kind of squint on what the ML

[00:07:18] researchers are doing later. I I think

[00:07:19] there's like

[00:07:21] it's possible to look at trends for

[00:07:22] sure. Recent trends, for example, seem

[00:07:24] to have been um

[00:07:26] hey look, Nvidia has a whole bunch of um

[00:07:29] control resources at all of the ASNs.

[00:07:31] Maybe we can figure out interesting ways

[00:07:32] to use them and do something more

[00:07:34] dynamic. So, like you can you can see

[00:07:36] people how they're responding to trends.

[00:07:38] I think long term you can't predict the

[00:07:40] model is going to be exactly this shape.

[00:07:41] That's hard to do.

[00:07:42] Um but we're going to use more of the

[00:07:44] resources that are available and less of

[00:07:45] the ones that aren't is is a pretty

[00:07:47] straightforward thing to predict, I

[00:07:48] would say.

[00:07:48] >> One of the things I've heard from

[00:07:50] researchers and um is sort of

[00:07:53] some of these chips that seem quite well

[00:07:55] co-designed, right? Uh they actually

[00:07:57] hate.

[00:07:58] Um so an example is like a TPU V6E, uh

[00:08:02] Trillium, I think is the public name.

[00:08:05] Uh Ghost Light is the name that everyone

[00:08:07] actually knows. Um but anyways,

[00:08:10] it's very common for people from

[00:08:12] DeepMind to be like, "Oh, this this chip

[00:08:13] is really weird cuz it's got like a ter-

[00:08:16] oh, sorry, a petaflop of compute and

[00:08:18] like 32 gigs of HBM. It's like quite a

[00:08:20] weird shape relative to the other chips

[00:08:23] of the time, right? It's got like 1/4

[00:08:25] HBM or something like that of a of a

[00:08:27] Hopper and only half the flops. And if

[00:08:29] you look at real utilizable flops, it's

[00:08:31] it's like the same, right? It's it's

[00:08:33] quite a weird chip. It makes sense in in

[00:08:35] many contexts, but I guess a lot of

[00:08:37] researchers hate it because it confines

[00:08:40] where they can do the research. Is there

[00:08:41] a way that co-design can go wrong?

[00:08:43] >> I mean, I kind of think you want to make

[00:08:45] a chip that people do hate to some

[00:08:46] extent. Like uh the there's

[00:08:50] people don't hate a chip if it's if it's

[00:08:52] got a very wide um the basin uh of of

[00:08:56] operating points where it can work in

[00:08:57] well. Um and I think that actually means

[00:08:59] you've left some resources on the table.

[00:09:01] So, uh the like really you're looking

[00:09:04] for

[00:09:05] like you would like your model to use

[00:09:07] all of the resources at 100% efficiency.

[00:09:09] Um 100% utilization. Um

[00:09:12] uh but if you can do that and sort of

[00:09:15] leave some slack around, then maybe you

[00:09:17] could have applied something. Often that

[00:09:18] is make a memory smaller because your

[00:09:19] batch size was too large. Or or if you

[00:09:21] have multiple different ways to lay out

[00:09:23] a model on on the chip, then um

[00:09:25] then probably there would have been one

[00:09:27] that was most efficient. Then the fact

[00:09:28] you could do other ones um is a missed

[00:09:30] opportunity. So,

[00:09:32] I mean,

[00:09:33] like I think it's a little glib to say

[00:09:35] that uh you want to use this to hate

[00:09:37] you, but uh but to some extent they

[00:09:38] should work pretty hard to actually uh

[00:09:40] get the most out of their hardware.

[00:09:42] >> I mentioned like sort of one that's from

[00:09:44] perhaps your history. Uh maybe one

[00:09:46] that's also from perhaps your history is

[00:09:47] like, you know, the the Dojo trips chips

[00:09:50] RIP, right? Um sad sad they're uh

[00:09:53] canceled and restarted maybe. I heard

[00:09:55] restarted. Um

[00:09:58] They they are also like really weird,

[00:10:00] right? In terms of like

[00:10:01] um

[00:10:02] You know, I guess I guess it comes back

[00:10:03] to my question of like do people hate

[00:10:05] this, right? Is Is you know, they're

[00:10:07] really really good at convolutional

[00:10:08] neural networks.

[00:10:09] Um there's like some really interesting

[00:10:11] data locality stuff, but then like you

[00:10:13] have no memory bandwidth, right? Like

[00:10:14] it's pretty bad. Um

[00:10:16] >> I mean, one of the one of the

[00:10:17] interesting constraints of chips is that

[00:10:19] they are two-dimensional. And you do

[00:10:23] actually have to confront this fact

[00:10:25] pretty actively when you're designing an

[00:10:27] ML chip. Um and that's why

[00:10:30] architectures like systolic arrays exist

[00:10:33] that are very naturally suited to a

[00:10:35] two-dimensional architecture, whereas

[00:10:36] you also see architectures

[00:10:39] um

[00:10:40] many architectures that have like an L2

[00:10:42] L1 cache that is

[00:10:45] that wants to be connected to all the

[00:10:46] different processors that are on the

[00:10:47] chip. And that's maybe less suited to

[00:10:51] uh embedding into 2D.

[00:10:52] >> Mhm.

[00:10:53] I guess but but in some way, right? Like

[00:10:56] you know, Tesla's a huge buyer of GPUs,

[00:10:58] right?

[00:10:59] Um they continue every, you know, 6

[00:11:01] months or so announce a big deal with

[00:11:02] Nvidia.

[00:11:03] Um of course they have their Dojo chips.

[00:11:05] Um

[00:11:06] So, we're going back to the whole point

[00:11:07] of like you've constrained the user a

[00:11:08] lot, right? Um

[00:11:11] Maybe Dojo's not good at certain kinds

[00:11:12] of model architectures, right? Uh is

[00:11:15] does that slow down Is there Is there a

[00:11:17] point where you co-design too much and

[00:11:19] now all of a sudden like, you know,

[00:11:20] you've left no flexibility on the table

[00:11:23] and you know, you you've got you burned,

[00:11:26] you know, the the architecture into the

[00:11:28] silicon and like you end up with like

[00:11:30] something that no one can use for modern

[00:11:32] stuff because ML research is moving way

[00:11:34] faster in different directions.

[00:11:36] >> Yeah, that's the challenge is that you

[00:11:38] have to

[00:11:39] as a hardware designer kind of predict

[00:11:40] the future. And you don't know where the

[00:11:43] future is going to lead. I think the

[00:11:46] most reliable way to think about this is

[00:11:49] just like when you're doing ML research,

[00:11:52] you are thinking about scaling loss and

[00:11:53] you're thinking about what is the

[00:11:57] how do I measure the direction that

[00:11:59] things are going so that I can launch a

[00:12:01] big run.

[00:12:02] And then now we're thinking about in

[00:12:03] terms of what is the direction things

[00:12:05] are going so that I can design a chip

[00:12:07] for

[00:12:08] where the models will be.

[00:12:09] >> And then the other thing you said is 2D,

[00:12:11] which makes sense on a chip level.

[00:12:12] Racks are three-dimensional though. So

[00:12:14] there's there's a point where that

[00:12:15] stops, right?

[00:12:15] >> That's true.

[00:12:16] >> Yeah.

[00:12:17] >> You can definitely come up with some

[00:12:18] very interesting network topologies in

[00:12:19] in these racks.

[00:12:21] There's the there's the torus for TPU

[00:12:24] famously. There's

[00:12:27] um

[00:12:28] Nvidia has a I guess you would call it a

[00:12:33] flattened butterfly?

[00:12:34] >> Isn't it just switched all tall?

[00:12:36] >> Yeah.

[00:12:37] >> [laughter]

[00:12:38] >> All sorts of fancy names for the same

[00:12:39] thing. Um

[00:12:41] Next generation TPUs have dragonflies.

[00:12:44] >> That's interesting.

[00:12:45] >> The traniums

[00:12:47] I don't know what the hell you'd call

[00:12:48] their architecture. It's like sort of a

[00:12:51] torus and not only that it switches but

[00:12:53] there's like switches in different trays

[00:12:54] and like you can do all talls kind of

[00:12:57] but like there's multiple it's weird. Um

[00:13:01] It makes sense. It makes sense though

[00:13:03] once you once you train up a model.

[00:13:04] Yeah, I guess I guess

[00:13:06] how does one like do you view it as just

[00:13:08] a spectrum of like I'm over optimizing

[00:13:10] for a problem or not or how much of this

[00:13:11] is like the art of just like

[00:13:14] you know

[00:13:15] I would I think models will be there.

[00:13:17] >> I mean I would say part of it is just

[00:13:18] you're providing the raw materials. Um

[00:13:21] you have to figure out like I mean

[00:13:23] there's sort of a whole spectrum of

[00:13:24] where you can provide these raw

[00:13:25] materials. If you provide um ingredients

[00:13:28] that are too small, like individual

[00:13:30] gates, that's what an FPGA is, it's 10x

[00:13:32] less efficient than an ASIC. So, don't

[00:13:34] do that. Um

[00:13:35] make the unit the grain size bigger. Um

[00:13:39] like systolic array as the grain size.

[00:13:42] Uh that's really pretty efficient for

[00:13:43] that point. Um Uh and so, pick the grain

[00:13:46] size appropriately. Um think about which

[00:13:48] fusions should be in place and are

[00:13:50] naturally there. So, maybe some

[00:13:52] operations always get fused together,

[00:13:53] and so you can have them as a bigger

[00:13:55] grain size. Um and then just try and

[00:13:57] expose those things uh

[00:13:59] with as little overhead as possible. Um

[00:14:01] so, I mean, that's sort of just like a

[00:14:04] hardware-centric approach to it. It's

[00:14:06] not really saying necessarily what are

[00:14:07] the models uh wanting and predicting

[00:14:09] what the models are, but it's just like

[00:14:10] the you can bound your downside by doing

[00:14:12] that. Like, I'm giving you all the

[00:14:14] things, maybe I don't give you as much

[00:14:16] of the things that I think you don't

[00:14:16] need, but mostly I'm giving you the more

[00:14:18] raw materials to work with. So, I think

[00:14:19] that's a pretty safe hedge. Um

[00:14:22] what you can do better than by taking

[00:14:24] into

[00:14:24] into account what you know about models

[00:14:26] is say, "Well, actually, um

[00:14:29] so, sure, we know that multiply and add

[00:14:31] should always be fused together into

[00:14:32] fused multiply add, uh but maybe I'll

[00:14:34] know something about um attention

[00:14:36] computations, and I can take advantage

[00:14:38] of the fact that there's always a

[00:14:39] softmax there, and I can do something

[00:14:40] special for that, and so on." And so,

[00:14:43] uh

[00:14:43] there's sort of a one angle on this is

[00:14:45] just say, um bound your regret on on the

[00:14:48] hardware design

[00:14:49] uh with as much information as you're

[00:14:50] willing to use about generally what I

[00:14:52] know about models.

[00:14:53] >> Well, one of the areas where

[00:14:56] co-design is a is a little bit

[00:14:58] contentious is um

[00:15:01] the claim that uh you can specialize a

[00:15:03] lot for transformers.

[00:15:04] >> Mhm.

[00:15:05] >> And

[00:15:06] maybe even burn in the architecture in

[00:15:08] some sense. Um

[00:15:11] what do you think of that of the of the

[00:15:13] claim that you can get

[00:15:14] 3x, 10x gains from specializing

[00:15:17] specifically for transformers?

[00:15:19] >> Pick the grains that you're working on.

[00:15:20] Um

[00:15:22] large matrix multiplies is a very clear

[00:15:25] one. Um

[00:15:26] like grain size continues to give you

[00:15:28] returns. It they diminish for sure, but

[00:15:30] um

[00:15:31] uh

[00:15:33] if you have lower precision matrix

[00:15:34] multiplies, they diminished like they

[00:15:36] have diminished less than than if it

[00:15:38] were uh

[00:15:39] just your

[00:15:40] the denominator is larger or it is

[00:15:42] smaller and so the the numerator becomes

[00:15:43] more important. Um

[00:15:46] uh and so I mean like what approach we

[00:15:48] take for example is just to say um

[00:15:51] matrix multiply and uh and some stuff in

[00:15:54] attention is is the place of

[00:15:55] specialization, um but like don't take a

[00:15:58] whole architecture. Do do appropriate

[00:15:59] size pieces.

[00:16:00] >> In

[00:16:01] uh you know, Ilya's words, right? We're

[00:16:02] in the age of research. Um you know,

[00:16:06] people are trying all sorts of crazy

[00:16:08] stuff.

[00:16:09] Um and I feel like 2 3 years ago, if you

[00:16:12] saw something that wasn't a transformer,

[00:16:14] at least personally, I ignored it.

[00:16:16] Um but now I see some of this like weird

[00:16:18] stuff and I'm like, "Oh, this is

[00:16:19] interesting."

[00:16:20] I still don't put much time into it, but

[00:16:22] like I I look at at least look at it. Um

[00:16:25] and if that's the case, right? You know,

[00:16:26] you know, as we move beyond transformers

[00:16:29] or at least you should have some as an

[00:16:31] as an AI lab, right? If you're not going

[00:16:32] to be an AI in Anthropic, GDM, you

[00:16:34] should at least have some portion of

[00:16:35] your compute into highly flexible uh

[00:16:40] compute, right? So so as an example, um

[00:16:43] TPUs aren't the best at like

[00:16:44] fine-grained like routing and dynamicism

[00:16:47] and

[00:16:48] um the programming model at least is

[00:16:50] pretty painful to do that, especially if

[00:16:51] you're just doing research and not like

[00:16:53] doing real training or inference, right?

[00:16:55] And so you see that with like you know,

[00:16:57] doing dynamic stuff with Palace and Jax

[00:16:59] and XLA is just a pain in the butt. Um

[00:17:02] and on the other hand, like GPUs cuz

[00:17:04] they have all these small cores, they're

[00:17:05] actually much better at this.

[00:17:07] Um and so Google still has some GPUs for

[00:17:09] DeepMind. Uh

[00:17:10] you think that's like a long-term state

[00:17:12] where every lab should have at least

[00:17:14] some budget of like, "Hey, screw it.

[00:17:16] Let's let's do some like weird wacko

[00:17:18] [ __ ] on CPUs too, right? Like

[00:17:20] >> I mean, it depends on how much room

[00:17:22] there is to optimize. Uh on the one

[00:17:25] hand, you can argue that

[00:17:27] transformers are kind of a

[00:17:29] they're they're a sufficiently general

[00:17:31] architecture that they're within a small

[00:17:33] constant factor of whatever possible

[00:17:35] thing you could possibly think of.

[00:17:37] Um

[00:17:39] or you could go the other way and be

[00:17:40] like

[00:17:41] well, the brain has neurons and

[00:17:43] something something about the amount of

[00:17:44] energy it takes to do something in a

[00:17:46] neuron. Um therefore, there's thousand X

[00:17:49] gains if we just change the

[00:17:51] architecture. And

[00:17:53] >> Well, this is what this is what the

[00:17:54] brain mouse company is doing, right?

[00:17:55] He's He's like He's like doing co-design

[00:17:56] from models all the way down to

[00:17:58] hardware. Everybody's doing like crazy

[00:18:00] analog [ __ ] right?

[00:18:01] >> So, if you're in a world where you think

[00:18:02] that things are going to happen like

[00:18:03] that, you should absolutely be looking

[00:18:06] at all this future model research and

[00:18:08] really focusing on that. And if you're

[00:18:10] thinking that things are going to be

[00:18:11] more stable from now on, let's just

[00:18:13] focus on

[00:18:14] increasing the grain size, uh making

[00:18:17] sure that we can really run what we

[00:18:19] think is important efficiently.

[00:18:22] And not focus so much on these like, I

[00:18:24] don't know, 1.2 X, 1.5 X gains where we

[00:18:27] could get more from the hardware.

[00:18:29] And I don't know which world we're in.

[00:18:32] >> I mean, you said Ilya I

[00:18:35] I don't know what's on his mind, but

[00:18:36] what it sounds like is on his mind is

[00:18:37] not necessarily all model architecture.

[00:18:39] There's a lot you can do

[00:18:41] loss function,

[00:18:42] um

[00:18:43] training data, and so on.

[00:18:45] Or or even just like take a transformer

[00:18:47] layer and then who knows, alternate it

[00:18:48] with something else. Or uh all of these

[00:18:51] things which are available within the um

[00:18:53] the world of um

[00:18:56] architectures that have like

[00:18:57] substantially had some of the fuss of

[00:19:00] transformers done at least. Um

[00:19:02] uh

[00:19:02] >> I I think it's funny by the way you said

[00:19:04] you don't know what Ilya is thinking. I

[00:19:04] think no one does.

[00:19:05] >> Yeah, right.

[00:19:06] Yeah.

[00:19:07] >> No one

[00:19:08] >> If you If you If you look at the like

[00:19:09] yeah, of course he does. Um and and

[00:19:10] these people do, but I think they're

[00:19:11] really close-lipped to the point where

[00:19:13] like, you know, SF gossip is usually at

[00:19:15] least somewhat okay about people and

[00:19:16] companies and things that are happening,

[00:19:18] but then like,

[00:19:19] you know, I've heard all the way from

[00:19:20] like they're making cyber weapons to

[00:19:21] like they're just trading financial

[00:19:23] markets.

[00:19:24] >> And that is.

[00:19:25] >> And they're doing everything in between

[00:19:26] and all those, yeah.

[00:19:28] >> Ilya is

[00:19:29] sees very far for sure, but uh

[00:19:32] like ultimately comes down to whatever

[00:19:35] he ends up shipping.

[00:19:37] >> Okay, so so like let's let's take take

[00:19:39] yourselves back to like 2022 is it when

[00:19:42] he was starting to write internally at

[00:19:43] OpenAI about reasoning, right?

[00:19:46] Um

[00:19:47] which eventually became strawberry and

[00:19:48] all these other things, right? Um

[00:19:50] you know,

[00:19:51] he he tried many many different things

[00:19:53] in this direction and finally, I think

[00:19:55] it was Yacob figured it out. I'm not

[00:19:57] sure if it was him or not.

[00:19:59] Um sorry if it was someone else, but you

[00:20:01] know, they they figured out like some

[00:20:03] form of, you know, test time scaling,

[00:20:06] you know, verifiable rewards, you know,

[00:20:09] this whole RL pipeline that has now been

[00:20:12] sort of giving us crazy gains over the

[00:20:14] last

[00:20:15] It's actually only been 14 months, by

[00:20:16] the way. Really? Since the first

[00:20:17] reasoning model.

[00:20:18] >> [laughter]

[00:20:19] >> Wow, that's been

[00:20:21] >> Anyways, uh it's only been like 14

[00:20:22] months.

[00:20:24] Is there something like,

[00:20:27] you know, you know, like if if if

[00:20:29] co-design is like, oh well, if you go

[00:20:30] back to 2022 everyone was thinking

[00:20:32] pre-training era, you know, just keep

[00:20:33] scaling pre-training. And now you're

[00:20:34] like, wait, actually Ilya was saying in

[00:20:36] 2022 it's not pre-training. Actually, we

[00:20:38] need to we need to co-design for

[00:20:39] sampling.

[00:20:40] Um what what would do you think like

[00:20:43] what do you think about co-design there?

[00:20:44] And like what what are your thoughts of

[00:20:45] like, hey,

[00:20:46] um what would you have done differently

[00:20:48] if you're like an Nvidia or someone else

[00:20:50] and in 2022 you decided this?

[00:20:52] >> Yeah, I think that the challenge there

[00:20:54] is that you can always have these

[00:20:56] paradigm shifts where suddenly sampling

[00:20:59] becomes so much more important

[00:21:02] to increasing intelligence in the model.

[00:21:04] So, you're no longer doing as much

[00:21:06] pre-training. You're now doing

[00:21:08] all of these uh uh these decode passes

[00:21:11] where your arithmetic intensity is very

[00:21:13] low. You need a ton of memory bandwidth.

[00:21:15] Suddenly, everything's changed for the

[00:21:17] hardware. Um

[00:21:19] and

[00:21:21] you can't really predict that as a

[00:21:23] hardware person.

[00:21:23] >> Wait, so if OpenAI made the chip in 2022

[00:21:25] when everyone was pre-training building

[00:21:27] only Ilya was like, "Let's sample a

[00:21:29] reasoning." then the chip would be in

[00:21:30] the wrong direction, right?

[00:21:32] >> You got to come in with with some amount

[00:21:34] of flexibility as to as to where you

[00:21:37] think things are going to go. And like

[00:21:39] at the time like Ilya was the only

[00:21:41] person who's up this. Eventually, it was

[00:21:43] two people, three people. And then

[00:21:44] eventually, suddenly the whole company

[00:21:46] and the whole industry.

[00:21:47] >> I think the labs have to hedge in this

[00:21:49] way. Like and I mean they have to hedge

[00:21:50] across different hardwares

[00:21:52] as a result. I think sort of the

[00:21:54] incentives for hardware vendors are

[00:21:56] maybe different in that

[00:21:57] um

[00:21:58] and and it really depends whether you're

[00:21:59] Nvidia where you can't fail. You have to

[00:22:01] make a generalist product like your next

[00:22:02] generation Nvidia product has to sell.

[00:22:04] >> Well, no, but they're they're making

[00:22:05] three products now, right? They're doing

[00:22:07] the they're doing the main line. They're

[00:22:08] doing the CPX, right? High arithmetic

[00:22:09] intensity. They're doing the like the

[00:22:11] Groq stuff, right? With it's like a You

[00:22:13] know, they're doing the 3D DRAM instead

[00:22:15] of like SRAM, but like you know,

[00:22:18] they're making different bets.

[00:22:19] >> That's Yeah, but there's some amount of

[00:22:20] that. So, if you can make multiple

[00:22:21] products, that's another way to hedge

[00:22:23] for sure. Um most companies that aren't

[00:22:25] Nvidia or maybe Google

[00:22:27] um

[00:22:29] can't spin off multiple products uh

[00:22:31] simultaneously. So, I mean I think the

[00:22:32] right thing that like for for companies

[00:22:34] that are making that choice is just like

[00:22:36] make the best guess and try. Um I think

[00:22:38] like if your best guess is

[00:22:40] like as making hardware trying to hedge

[00:22:42] across all all possible scenarios, um

[00:22:45] you just don't win. Um like you don't

[00:22:47] win any in any of the scenarios. Um

[00:22:50] you totally can as a lab because you can

[00:22:52] wait and see and and decide which one to

[00:22:53] buy. But but as a as a hardware vendor,

[00:22:55] you have to make some some some effort

[00:22:57] for sure.

[00:22:58] >> So so I guess I guess I'm curious of

[00:22:59] your two perspectives, right? Like

[00:23:02] you know, Cerebras, Groq, they were

[00:23:03] working on something that

[00:23:05] and and they were not focused on the

[00:23:07] low-latency sampling regime, I don't

[00:23:09] think. Um Yeah, I think similar

[00:23:11] >> It seems like they kind of fell into

[00:23:12] that accidentally, where they were

[00:23:14] >> it's now a co-design point that actually

[00:23:16] makes it

[00:23:16] >> Exactly, exactly. They it it worked out

[00:23:19] so well because they were working on

[00:23:21] these ConfNets that were very small and

[00:23:23] could fit entirely on

[00:23:25] >> The other picture here is it's LSTMs

[00:23:27] that were also like very

[00:23:29] >> uh small, very very uh sequential.

[00:23:32] >> Yeah.

[00:23:33] >> Yeah, and so so you can fit that all

[00:23:34] into one tile. Same thing for for for

[00:23:36] Dojo as well. Uh and then suddenly LLMs

[00:23:40] come along. It actually turns out you do

[00:23:42] need to be able to have like terabytes

[00:23:45] of parameters now. Uh

[00:23:48] and that's a very different regime, but

[00:23:51] in this specific market where you really

[00:23:54] really care about low latency above all

[00:23:56] else, like like encoding

[00:23:58] uh like in quite a few um

[00:24:01] of these like thinking models would just

[00:24:03] just take too long. Um

[00:24:06] this actually really matters and this is

[00:24:07] a really good niche for them to be in.

[00:24:09] >> So, I mean my take at least on on on the

[00:24:11] low-latency thing is uh I mean, if you

[00:24:14] look at what that means, you are

[00:24:16] um I mean, you want low latency, for

[00:24:18] sure. Uh and SRAM is really the only

[00:24:20] only strategy to get there. Maybe the 3D

[00:24:22] stacked DRAM is is um isn't a perfect

[00:24:25] way to get it close to that. Um it's

[00:24:27] it's interesting for other reasons, like

[00:24:28] KB hash. Um But then, what do you do

[00:24:30] with all of that compute? Like you made

[00:24:32] a you've made a product that has a lot

[00:24:33] of compute because of the other use

[00:24:34] cases for prefill and training and so

[00:24:36] on. Um

[00:24:37] and I mean, if you look at lab pricing,

[00:24:40] it kind of looks like the fact just the

[00:24:43] fact that prefill tokens are I don't

[00:24:44] know, five times cheaper than decode

[00:24:45] tokens

[00:24:47] says something about what utilization or

[00:24:49] efficiency is in prefill versus decode.

[00:24:51] That probably means that during decode

[00:24:53] you have a whole bunch of compute

[00:24:54] sitting idle.

[00:24:56] I think like the biggest uh research

[00:24:58] agenda for taking advantage of that is

[00:25:00] to say, "Hey, can I take a model and

[00:25:02] uh make uh the um the MLP much bigger?"

[00:25:06] And somehow use those spare flops. Like

[00:25:08] 80% of your flops are sitting idle.

[00:25:10] Do anything at all it must be better

[00:25:11] than doing nothing and and and take

[00:25:13] advantage of that. Um

[00:25:16] I mean, simplest thing to do is just

[00:25:17] crank up the MLP size by a factor of

[00:25:19] five and and take advantage of that.

[00:25:21] >> We're in an age where you see a lot of

[00:25:23] disaggregation, right? Um people are

[00:25:25] doing uh disaggregation of prefill and

[00:25:28] decode. Initially, they were doing on

[00:25:29] the same chips. Um doing things like

[00:25:31] chunk prefill as well. Anything to

[00:25:32] anything to sort of like get utilization

[00:25:34] up higher. Uh then they started doing

[00:25:35] disaggregated prefill decode on

[00:25:36] different kinds of chips, right? Um at

[00:25:39] least that's what's Nvidia's pitch

[00:25:41] today.

[00:25:42] Um

[00:25:43] and then they're sort of like we we you

[00:25:44] know, we've at least built our own

[00:25:46] shitty simulator which is probably

[00:25:47] internally which probably worse than

[00:25:49] yours and worse than yours. Um but at

[00:25:51] least it's like given us some ideas on

[00:25:53] like, "Hey, actually you would want to

[00:25:55] You may want to disaggregate the MLP

[00:25:57] from the attention even, right?" And MLP

[00:25:59] you could have the weights in let's say

[00:26:01] SRAM or 3D RAM and the attention you

[00:26:04] want through HPM. And this is something

[00:26:05] that we've also like sort of like I

[00:26:07] guess

[00:26:08] what where do you do you see this wave

[00:26:10] cuz you're both you know, you're making

[00:26:11] one chip, right? Like at least uh my

[00:26:13] understanding is you're making one chip.

[00:26:15] Um there's a wave of like different

[00:26:17] optimization points for prefill and

[00:26:18] decode. Um

[00:26:20] you know, say if in a sampling heavy

[00:26:22] regime versus in a regime where you're

[00:26:24] doing a lot of backwards passes. Um

[00:26:26] >> Yeah, I I think I think it's really cool

[00:26:29] to think about decoupling uh where for

[00:26:32] example, sparse uh sparse MLPs where you

[00:26:36] suddenly are decoupling your the amount

[00:26:38] of math you have to do and the amount of

[00:26:42] a kind of effective model capacity for

[00:26:44] knowledge. Um and this really affects

[00:26:48] the hardware in a lot of ways because

[00:26:49] now you're not you're not tied together

[00:26:51] like that anymore. Now, you can

[00:26:53] have much less arithmetic intensity for

[00:26:55] the same amount of HBM, uh same amount

[00:26:57] of HBM bandwidth uh for reading in the

[00:27:00] parameters. And so, decoupling the

[00:27:03] attention in the the MLP is very

[00:27:05] interesting because these are very

[00:27:07] different operations. Uh and if you do

[00:27:10] put these on different devices,

[00:27:12] uh maybe even like totally different

[00:27:14] memory technologies even, uh like you're

[00:27:16] suggesting, I think you can have a lot

[00:27:19] of wins.

[00:27:20] Uh

[00:27:20] it is

[00:27:22] maybe a little bit scary from the

[00:27:23] software perspective. Uh

[00:27:25] now, you have to coordinate two

[00:27:26] different types of devices.

[00:27:28] Um

[00:27:29] Who knows what's going to happen there?

[00:27:31] >> Some of the like this the thing that

[00:27:32] makes it scary is you have to decide um

[00:27:35] somewhat a a priori of uh how many

[00:27:38] resources are going on this side and how

[00:27:39] many resources resources going on that

[00:27:40] side. That's true. Um

[00:27:42] and so, uh that ends up baking in this

[00:27:45] is going to be my ratio of attention to

[00:27:46] MLP or to MLW or something like that. Um

[00:27:50] and I mean, we we we make those

[00:27:51] decisions all the time. We pick on

[00:27:53] resource balances between memories and

[00:27:54] and and compute and so on. Um this is

[00:27:57] one more of them. Um and it's sort of

[00:27:59] like uh

[00:28:00] and sort of doubling all the decisions I

[00:28:01] make I make all the decisions on the

[00:28:02] left and then I make all the decisions

[00:28:04] on the right.

[00:28:05] To some ex- and and then I guess the

[00:28:06] other thing that you can get like if the

[00:28:08] the trade-off is also removes your

[00:28:11] ability to steal um dynamically uh like

[00:28:14] uh uh re- steal resources between

[00:28:17] attention and and and and and Emily. So,

[00:28:21] uh a classic thing is to um be fetching

[00:28:25] always be fetching KBs from HBM um even

[00:28:28] while you're running the MLE. Um

[00:28:30] uh and and then use a little bit of the

[00:28:32] HBM bandwidth to fetch weights or

[00:28:33] something as up like that as well. Um

[00:28:35] and so, uh sliding where that that that

[00:28:38] divider is is is something you can do on

[00:28:39] a single chip and it's difficult to do

[00:28:40] on multiple chips. As the workloads

[00:28:42] become more expensive for sure, like the

[00:28:45] um

[00:28:46] just find every little part and make the

[00:28:48] the thing you can for those little

[00:28:49] parts. Seems like a long time trend.

[00:28:50] It's just like when is the right time is

[00:28:52] is is the question.

[00:28:53] >> So, one of my pet peeves in quantization

[00:28:56] papers is that

[00:28:58] uh they come out with a claim that says

[00:29:00] we're 97% as accurate as

[00:29:03] uh as the full thing. And turns out if

[00:29:05] you look into the numbers, actually this

[00:29:07] is like downgrading a 70 billion model

[00:29:09] to an 8 billion model.

[00:29:11] Um, how do you think about this?

[00:29:12] >> Yeah, what what does accurate even mean,

[00:29:13] right? Like you take perplexity and it's

[00:29:15] like

[00:29:16] one over 97% of perplexity or something

[00:29:19] like that. Or like 97% uh MMLU score or

[00:29:23] something like that. Yeah.

[00:29:24] >> I It's It's 97% of the MMLU score once

[00:29:27] you did post-quantization

[00:29:29] training. You know, that's usually what

[00:29:31] they do.

[00:29:32] >> Like you see these companies advertising

[00:29:34] like these 0.1%

[00:29:36] differences on these

[00:29:38] uh on these evals.

[00:29:39] >> So, I mean, the problem behind this,

[00:29:41] right, is the

[00:29:43] um

[00:29:44] >> Actually, I don't know if it's a

[00:29:45] logarithmic relationship between amount

[00:29:47] of compute and and like perplexity

[00:29:49] improvement or something like that.

[00:29:50] Uh so,

[00:29:53] so that's what gives you all of this

[00:29:54] sensitivity to those small things.

[00:29:56] Um,

[00:29:57] why not just hit the same quality with

[00:29:58] more model parameters? That seems a much

[00:29:59] fairer way. So, um

[00:30:02] I think that it should be a standard for

[00:30:03] how how

[00:30:04] how you do these things. We

[00:30:06] in general, we find that we need to

[00:30:08] increase the model size by a 40% in

[00:30:10] order to hit the same quality level. Um,

[00:30:13] a better paper would be 35% instead of

[00:30:14] 40%.

[00:30:16] >> Yeah, makes sense.

[00:30:17] The way I like to frame it is perplexity

[00:30:19] per picojoule.

[00:30:21] Where

[00:30:22] if you are the same perplexity,

[00:30:24] how many picojoules how much energy did

[00:30:26] you take to generate this token?

[00:30:28] >> Per picojoule. So, if I have twice as

[00:30:30] many picojoules, I get twice the

[00:30:32] perplexity?

[00:30:33] >> That's not literally division. It's a

[00:30:35] There's a predo there.

[00:30:36] >> Okay.

[00:30:36] >> What is it your license plate? Can you

[00:30:38] tell us your license plate?

[00:30:40] >> It's exaflop.

[00:30:41] >> It's exaflop. Oh, I thought it was like

[00:30:42] intelligence per picojoule or something

[00:30:44] like this.

[00:30:45] PJ p e r o p picojoules per op is open.

[00:30:50] It is open source. Oh, wow.

[00:30:53] PJ per p per bit, too, would be pretty

[00:30:55] good, but that is that's too many

[00:30:56] letters, eh?
