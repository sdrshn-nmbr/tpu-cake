[00:00:01] [music]

[00:00:06] [music]

[00:00:06] >> So, Bernard, I don't know. It's 2027.

[00:00:09] What what do chips look like? What's

[00:00:10] what's different? I mean, 2027 is

[00:00:12] close-ish, uh at least on chip

[00:00:14] timelines. Uh maybe not on AI timelines,

[00:00:16] but for sure on chip timelines.

[00:00:18] >> Those are Those are on a week-by-week

[00:00:19] basis. Yeah, I know, right?

[00:00:20] Um so, I mean, that is really sort of

[00:00:23] answering the question of we're like,

[00:00:24] what are the chips that have been in

[00:00:25] development in the last year or two? Um

[00:00:27] uh

[00:00:29] I think the

[00:00:30] like what are the big physics trends,

[00:00:31] right? Like uh

[00:00:33] flops are cheap. Um we can still pack

[00:00:35] more and more of them into a chip uh

[00:00:36] into the same amount of logic. Um uh

[00:00:39] power density is going up as well. Like

[00:00:41] we can just deliver more power to the

[00:00:43] chips um and take the heat out. Um

[00:00:46] and then uh just figure out how to use

[00:00:49] that SRAM. Those are I think the big

[00:00:50] three opportunities that we um

[00:00:53] like is just like taking existing

[00:00:55] technology and bringing it to to like

[00:00:56] actual product market fits now. So, I

[00:00:58] think that's in the short term is what

[00:00:59] we should expect to see chips that um

[00:01:01] are like many chips that are as low

[00:01:03] latency as Groq and Cerebras. Um that's

[00:01:05] through weights and SRAM.

[00:01:06] Um

[00:01:08] There's a whole thing there of like

[00:01:09] weights and SRAM but probably KVs in

[00:01:10] HBM.

[00:01:11] Um

[00:01:14] And then more and more flops just

[00:01:15] because the sort of the What is the

[00:01:18] exchange rate between like uh a marginal

[00:01:20] flop versus a marginal um

[00:01:22] uh

[00:01:22] byte per second of HBM bandwidth? Uh the

[00:01:25] sort of the cost of more flops is is is

[00:01:26] a lot cheaper. And so, uh you can figure

[00:01:28] out how to use that better. What what

[00:01:30] what does the ratio look like right now?

[00:01:32] Yeah, so

[00:01:33] like I guess on the HBM side um

[00:01:36] a a marginal uh byte per second of HBM

[00:01:38] bandwidth really just means like I need

[00:01:40] to buy another stack of HBM. There's not

[00:01:42] really any way I can sort of uh I may I

[00:01:44] make I can maybe crank up the frequency

[00:01:46] on on on the on the pin speed. Um but

[00:01:50] beyond that I just need to buy more

[00:01:51] stacks of HBM. HBM is a big fraction of

[00:01:53] the um sort of total cost of ownership

[00:01:56] of of a chip. Um in in many cases it can

[00:01:59] actually be more expensive than the the

[00:02:01] logic die. Um

[00:02:03] So, like and we're really up against the

[00:02:05] limits of of of the physics on on the

[00:02:07] memory bandwidth side. So, like next

[00:02:09] generation HBMs like there's even

[00:02:10] question of can we cool it? Do we need

[00:02:12] to separate it from the logic die to be

[00:02:13] able to cool it? Um big physical uh uh

[00:02:16] challenges. Um whereas for logic uh

[00:02:20] we

[00:02:21] to some extent Moore's law is still

[00:02:23] happening. It's a lot slower than it was

[00:02:24] in the past, but there are improvements

[00:02:26] generation over generation. And then

[00:02:27] architecturally there's just big things

[00:02:28] you can do to pack in um still another

[00:02:31] factor of two, three, four more flops um

[00:02:33] into logic dies. And so, uh there just

[00:02:35] seems to be for the time being and for

[00:02:37] the last few years like a lot more

[00:02:38] headroom to cheaply increase flops.

[00:02:40] Right. Let's talk about cooling here.

[00:02:42] Like um I imagine with more and more

[00:02:44] flops going into chips we're probably

[00:02:45] looking at water cooling.

[00:02:47] Yeah, for sure. Uh so, yeah. Um

[00:02:50] Yeah, how how do you square with this

[00:02:52] with like I don't know. Like it seems

[00:02:53] significantly harder and more expensive

[00:02:54] to build out the system now. Um how are

[00:02:56] you sort of thinking of of that

[00:02:58] trade-off relative to flops when it

[00:02:59] comes to like sort of like the the total

[00:03:01] cost of ownership here?

[00:03:03] Yeah, so I mean, I think it really

[00:03:04] depends on what's the right denominator,

[00:03:05] right? So, if you look at the cost of a

[00:03:07] system, like what is the cost of a rack?

[00:03:08] That has been going up and up and up.

[00:03:10] But the rack does like twice as much or

[00:03:11] three times as much or something like

[00:03:12] that. So, it's like the right

[00:03:14] denominator is cost per flop, cost per

[00:03:17] token per second, something like that.

[00:03:19] Um

[00:03:20] and uh

[00:03:22] the reason you go to higher power is

[00:03:24] that you can also go to higher clock

[00:03:26] frequency on your chip. Um and so, for

[00:03:29] the same square millimeters of silicon

[00:03:31] you can get twice the performance or

[00:03:32] three times the performance or something

[00:03:33] like that. So, and and you don't pay

[00:03:35] twice or three times as much on on the

[00:03:37] on the cooling uh there. Um so, that

[00:03:40] ends up being a net win. Um and I think

[00:03:43] probably keeps pushing for a while.

[00:03:45] I mean, I know this maybe sort of

[00:03:47] harkens back to like the

[00:03:49] Pentium era like pushing 4 GHz clock

[00:03:51] speeds on CPUs, but there's sort of a a

[00:03:54] difference there, which is that it's um

[00:03:56] in those days it was like clock speed is

[00:03:57] how you get the sequential time. Like

[00:03:59] the number of sequential instructions um

[00:04:01] to improve, whereas today it's how do

[00:04:03] you get like more parallel uh throughput

[00:04:06] out. And so, that is in some sense a

[00:04:08] more um

[00:04:10] uh you're not as much beating your head

[00:04:12] against the wall to uh there. Like it's

[00:04:14] actually productive. Yeah, I remember

[00:04:15] like trying to overclock uh CPUs back in

[00:04:17] the day and you have to put these like

[00:04:18] crazy crazy liquid coolers on there and

[00:04:20] Yeah, and just on the architecture side

[00:04:22] like they had super deep pipelines. Like

[00:04:23] I think the the extreme was like a

[00:04:25] 15-stage CPU pipeline, which means your

[00:04:27] branch misprediction penalties 15

[00:04:29] cycles. Um

[00:04:32] Branch mispredict pallet penalties on

[00:04:34] like AI chips are like can be hundreds

[00:04:36] of clock cycles and it's fine cuz you

[00:04:38] don't take branches that often. And so,

[00:04:39] you can do like like

[00:04:41] millions or billions of multiplies in in

[00:04:43] that time and it's totally fine. On on

[00:04:45] on sort of the architecture side I think

[00:04:46] we should maybe go back and talk about

[00:04:47] the the SRAM. Like what's what's what's

[00:04:49] new here, right? Like um I think like

[00:04:51] there's a lot of interest in sort of low

[00:04:52] latency high IOPS chips now. Um

[00:04:55] and like part of the MatX claim here is

[00:04:56] that you can just like do that and also

[00:04:58] deliver good throughput, right? Like

[00:05:00] Let's talk about how.

[00:05:01] Yeah, okay. So, um

[00:05:04] So, firstly like SRAM is on every single

[00:05:06] chip always. And so, like when we say

[00:05:08] what is an SRAM chip? Like why why is uh

[00:05:11] like Google's TPUs, Amazon's Trainium,

[00:05:13] Nvidia, why they're not SRAM chips? Um

[00:05:16] Firstly, SRAM means weights in SRAM. Uh

[00:05:19] and so, that is what Groq and Cerebras

[00:05:20] have been doing.

[00:05:21] Um

[00:05:22] and uh there is sort of a relationship

[00:05:25] of in order to support weights in SRAM I

[00:05:26] need

[00:05:28] um

[00:05:28] uh I need a combination of enough SRAM,

[00:05:31] enough interconnect to support all the

[00:05:32] tensor parallelism, expert parallelism

[00:05:34] to get the data in and out um and the

[00:05:37] how that relates to the number of flops

[00:05:38] on the chip. So,

[00:05:40] when we say an SRAM chip, we really mean

[00:05:42] all of that entire package is

[00:05:44] uh configured correctly so they can I

[00:05:46] can um keep my weights long-term

[00:05:48] resident in SRAM.

[00:05:49] Um

[00:05:50] and that genuinely just does give you a

[00:05:52] lot better latency than than HBM. Um

[00:05:54] latency of a model goes as really like

[00:05:58] it's the time it takes to load the model

[00:06:00] weights into the multipliers. Um and so,

[00:06:02] that's just like number of parameters

[00:06:04] divided by the bandwidth. SRAM bandwidth

[00:06:06] is a hundred times uh higher than HBM

[00:06:09] bandwidth. And so, you can get a lot

[00:06:10] lower latency. Um you tend to get only

[00:06:12] 10 times lower latency, not a hundred

[00:06:13] times lower, but uh ballpark you can get

[00:06:16] uh about an order of magnitude better.

[00:06:17] That's the SRAM side. Um the the reason

[00:06:20] that Groq and Cerebras as standalone

[00:06:23] products have struggled is that

[00:06:25] uh

[00:06:27] SRAM capacity is so small and um and so,

[00:06:31] like your challenges are can I fit the

[00:06:33] weights? Can I fit the KVs? Mhm.

[00:06:35] Can I fit the weights? Uh that actually

[00:06:37] works totally fine. Run a pipeline

[00:06:39] across hundreds of chips, thousands of

[00:06:40] chips. Um

[00:06:42] Put a tiny fraction of the weights in

[00:06:43] every chip. Um you can totally do that.

[00:06:46] Uh you pay almost no penalty in in any

[00:06:48] scaling term. You have to pay for the

[00:06:50] interconnect for it. The interconnect

[00:06:51] isn't that bad.

[00:06:52] Um

[00:06:53] uh and so, uh you can unroll these very

[00:06:55] deep pipelines.

[00:06:58] Can we do the same strategy for KVs? We

[00:07:00] can't. So, the problem is that when when

[00:07:03] you unroll uh KVs over this huge number

[00:07:05] of chips, you don't actually get any net

[00:07:07] saving because

[00:07:09] uh

[00:07:09] now your batch size has grown by the

[00:07:11] pipeline depth. So, batch size is n

[00:07:13] times larger. You've saved n times uh

[00:07:15] capacity per chip, but that cancels out

[00:07:17] and you actually don't get any saving at

[00:07:18] all.

[00:07:19] >> [snorts]

[00:07:19] >> So, um SRAM basically as a result, KVs

[00:07:23] in SRAM does not work. There's not

[00:07:24] enough capacity. You need to put them in

[00:07:25] HBM.

[00:07:26] So, I think

[00:07:29] what that ends up meaning is the um

[00:07:32] you need a chip with enough SRAM that

[00:07:34] you can do a weights long-term in in

[00:07:35] SRAM, but then also put your KVs in HBM.

[00:07:38] Um and so, we're seeing some signs of

[00:07:41] that in the market already. The um

[00:07:43] >> [clears throat]

[00:07:44] >> This is how

[00:07:45] uh Nvidia and Groq are starting to

[00:07:47] deploy there um hybrid like put a Nvidia

[00:07:50] rack next to a Groq rack. Um

[00:07:53] Really I think the long term is that you

[00:07:55] integrate it much more tightly and you

[00:07:56] put uh a lot of SRAM and a big HBM um in

[00:07:59] the same package. And that gives you

[00:08:00] much power. Mhm. Yeah, like in in the

[00:08:02] Nvidia and Groq case I guess these are

[00:08:03] like two separate systems right now.

[00:08:04] Like

[00:08:05] I guess what's the trade-off here of

[00:08:07] like I don't know. Sort of a

[00:08:11] you're doing, right? Like a a chip that

[00:08:12] just has has both a lot of SRAM and and

[00:08:15] a lot of HBM.

[00:08:16] Yeah, so I mean,

[00:08:18] what it like sort of the trade-off is

[00:08:21] you've put this I mean, for for

[00:08:22] understandable reasons and it's a good

[00:08:23] like it's a good sort of practical

[00:08:25] measure to do what they've done. Um but

[00:08:27] what it does is it puts this puts this

[00:08:28] barrier in between. And so, I've got a

[00:08:30] certain number of resources on on the

[00:08:31] left and a certain number of resources

[00:08:32] on the right.

[00:08:34] And you're stuck with that split. And if

[00:08:36] you want to move stuff from one side to

[00:08:38] the other, like I want to move for

[00:08:39] example some of my flops from the Nvidia

[00:08:42] chip to the Groq chip or something like

[00:08:43] that. Um

[00:08:46] How do you do that? Like you're you're

[00:08:47] stuck. You have to move uh the

[00:08:48] computation there, which means you have

[00:08:50] to move the uh interconnect there. And

[00:08:52] and so, you have this interconnect

[00:08:53] bottleneck then.

[00:08:54] You can just tightly couple the HBM into

[00:08:57] the SRAM if you design a a system for

[00:08:58] that. Um and that gives you all of that

[00:09:02] gives you the opportunity to move all of

[00:09:03] the HBM bandwidth into SRAM. Um and that

[00:09:06] actually gives you a lot of interesting

[00:09:07] new opportunities. One example would be

[00:09:10] uh that you can um do a a load of

[00:09:14] something, typically a KV cache, um from

[00:09:16] HBM into SRAM and then keep it in SRAM

[00:09:18] for quite a long time. Uh but then bring

[00:09:19] the next one in. And so, that's

[00:09:21] uh just like this is sort of if you were

[00:09:23] designing from first principles, that's

[00:09:25] what you'd do. Right. Yeah. And in terms

[00:09:27] of like what you're designing these

[00:09:28] chips for, right? Let's talk a little

[00:09:30] bit about that. Like

[00:09:32] I think you're like being fairly

[00:09:33] targeted in like what sort of models

[00:09:34] you're trying to support. How are you

[00:09:35] reasoning about this? How are you sort

[00:09:37] of like thinking ahead to like when

[00:09:38] these chips are are rolling out, like

[00:09:39] what are people going to want to train

[00:09:40] and and run? Yeah. Um so, I mean, the

[00:09:44] the original vision was like uh the

[00:09:46] needs of Frontier Labs. That is still

[00:09:47] the vision. Um

[00:09:49] And and and I think that mostly shows

[00:09:52] shows up as very large models, very

[00:09:53] large batch sizes. Um

[00:09:56] Still low latency but large batch sizes.

[00:09:58] Um I think it's maybe worth contrasting

[00:10:01] that with

[00:10:02] like

[00:10:04] approaches that other chips have taken

[00:10:05] in the past.

[00:10:06] One of the big things was like saying

[00:10:08] can we do like batch size one, can we do

[00:10:10] small models like convolutions or things

[00:10:11] like that, things with small channel

[00:10:13] counts.

[00:10:16] I have some past experiences working on

[00:10:17] such chips as well.

[00:10:20] One of the challenges there is that you

[00:10:22] spend a huge amount of the chip, I mean

[00:10:24] like actually just thinking time in how

[00:10:26] you design the chip but as well as

[00:10:28] area and silicon on the chip supporting

[00:10:31] these small models well and it's just a

[00:10:32] much harder problem. You have to figure

[00:10:33] out how to take like a small convolution

[00:10:36] kernel and replicate it over all of your

[00:10:37] chip or and then solve the data movement

[00:10:40] problems inside the chip that are

[00:10:41] associated with that.

[00:10:43] Or

[00:10:45] manage my low batch size and figure out

[00:10:47] how to extract all of the possible

[00:10:49] parallelism from from from a single user

[00:10:51] rather than extracting parallelism from

[00:10:53] just like I've got 10,000 users in

[00:10:55] concurrently.

[00:10:57] So that was a whole bunch of no we don't

[00:10:58] want to do that stuff.

[00:11:00] Nvidia has always been supporting that

[00:11:01] stuff as as well as they can. A lot of

[00:11:03] the 2017 era of startups like did it put

[00:11:06] a lot of the innovation into that

[00:11:07] problem and we said no no thank you not

[00:11:09] to that.

[00:11:10] We are going to have the whole chip

[00:11:11] focusing on one

[00:11:13] sort of coherent thing at a time.

[00:11:17] And rely on the fact that we have enough

[00:11:19] work to do, large enough matrices,

[00:11:21] enough large enough batch sizes that we

[00:11:24] can have a little chip doing that one

[00:11:25] thing at a time. That leads to a much

[00:11:28] simpler design.

[00:11:31] We then spend some of that

[00:11:33] simplicity

[00:11:34] to then like use our innovation budget

[00:11:37] on

[00:11:38] okay can we support attention with this

[00:11:40] like small attention head size really

[00:11:42] efficiently for example.

[00:11:44] So so overall I think that ended up

[00:11:46] being

[00:11:49] LLMs in the way we see them now and and

[00:11:51] evolving.

[00:11:53] I think those are all the sort of the

[00:11:54] qualitative architectural sides of it

[00:11:56] and then the product choices are

[00:11:58] low latency is like it's just a free

[00:12:00] like it's a free idea that

[00:12:04] needs to be put into the right product

[00:12:05] which is SRAM and HBM and then put a

[00:12:08] huge amount of flops in because like

[00:12:09] flops is one of the most valuable

[00:12:11] resources. Yeah.

[00:12:13] And and so like

[00:12:15] with all these with all these flops like

[00:12:16] how how are you thinking about like I

[00:12:18] don't know like how are people going to

[00:12:19] use these flops?

[00:12:20] I you have an ML team that thinks about

[00:12:23] things like this. Let's let's chat about

[00:12:24] like some of the work they're doing. So

[00:12:26] I mean let's start with

[00:12:28] how do you use those flops at least in

[00:12:29] the first place. So training prefill

[00:12:32] decode training and prefill are

[00:12:34] compute limited. You can always apply

[00:12:36] more flops there it's very pretty

[00:12:37] straightforward.

[00:12:38] Um

[00:12:39] The sticker is is decode.

[00:12:43] The sort of the common

[00:12:46] concern or hesitation is

[00:12:48] decode is entirely HBM bandwidth

[00:12:49] limited. There's nothing you can do so

[00:12:52] like don't don't try to apply more flops

[00:12:54] there. There's a few angles to say that

[00:12:56] that might be

[00:12:57] wrong or at least short-sighted.

[00:12:59] Um

[00:13:00] Maybe one of the most fundamental is

[00:13:02] like

[00:13:04] if you've got a chip where the flops are

[00:13:06] sitting idle surely there must be

[00:13:07] somewhere you can figure out how to use

[00:13:09] them productively.

[00:13:10] And I I know it's kind of a little glib

[00:13:12] because like

[00:13:13] um

[00:13:14] like let's talk about the specific ways

[00:13:15] rather than just like abstractly what

[00:13:17] the what the what they could be.

[00:13:18] But there's a big part of

[00:13:20] like I think like I'm sure many many of

[00:13:24] the labs in general are going to be

[00:13:25] working on on answering this exact

[00:13:27] problem but

[00:13:29] we have some opinions on like maybe it's

[00:13:30] actually existential dramatics as well

[00:13:31] and so

[00:13:33] like thinking about what that might look

[00:13:34] like a few years from now. That's a big

[00:13:36] part of what our ML team does.

[00:13:39] So to break down that problem like the

[00:13:42] in in a model you have the attention

[00:13:44] layer and the feedforward network layer

[00:13:45] or the MoE layer.

[00:13:47] You can always throw more flops at

[00:13:49] feedforward network MoE.

[00:13:51] Just make it bigger,

[00:13:52] make it less sparse,

[00:13:54] run it a few times in a row, all kinds

[00:13:56] of things like that.

[00:13:58] What is the most productive way to to

[00:14:00] make those changes and actually realize

[00:14:01] that in terms of model quality? And at

[00:14:03] some point like if you can make that big

[00:14:04] enough it'll like you should get to a

[00:14:05] balance point between

[00:14:07] memory bandwidth and and flops.

[00:14:09] The other side of that so that's make

[00:14:11] the MoE bigger. The other side of that

[00:14:13] is make the attention smaller. So

[00:14:16] there's so many things in the literature

[00:14:17] for that. Generally general ideas that

[00:14:20] are well established are like

[00:14:22] alternating long and short context,

[00:14:24] reusing KB caches over consecutive

[00:14:26] layers, using fewer KB heads,

[00:14:29] and then maybe things like multi-head

[00:14:30] latent attention where you sort of

[00:14:31] compress the heads down and then grow

[00:14:33] them up.

[00:14:34] And then maybe even like

[00:14:36] a bigger hammer than all of that is to

[00:14:37] access the KB cache sparsely.

[00:14:39] So a range of all of those things there

[00:14:41] are so many ideas to grapple with there

[00:14:43] and like what is the right way to think

[00:14:44] about that? One of the things we we do

[00:14:47] on our ML team is just to

[00:14:49] try and understand the space and then

[00:14:50] quantify what is the exchange rate

[00:14:52] between HBM bandwidth and

[00:14:55] and and flops. So

[00:14:57] how is there even an exchange rate

[00:14:59] there? Really that's what saying is like

[00:15:02] let's say I want to get 1% improvement

[00:15:04] in

[00:15:06] in model quality in some sense. So like

[00:15:08] as if you know as if I had double the

[00:15:09] size of my model something like that. So

[00:15:11] um

[00:15:12] we know how to do that on flops.

[00:15:14] Keep doubling the size of the model

[00:15:15] until the quality is better. How many

[00:15:17] doublings of HBM bandwidth do I need to

[00:15:18] get the equivalent?

[00:15:20] Uh

[00:15:22] And so like just a starting point is

[00:15:24] just measure that and understand where

[00:15:25] it's at.

[00:15:27] Really interesting problem like

[00:15:29] there's many difficult ways to measure.

[00:15:31] Do you have to be very sensitive to how

[00:15:32] you measure it right because

[00:15:34] simple ways of measuring like loss for

[00:15:36] example if you just look at the impact

[00:15:38] on loss it'll say actually you know just

[00:15:41] like have short context of 64 tokens

[00:15:43] never attend beyond that and just have a

[00:15:45] massive model and that actually is what

[00:15:47] the like that'll optimize the loss for

[00:15:50] you know per cost in in some sense.

[00:15:53] So you know some of what we spend some

[00:15:54] time being maybe we can be a little bit

[00:15:56] more sophisticated about how you measure

[00:15:59] quality and like actually sort of up

[00:16:01] weight long context

[00:16:03] like aspects of quality that show up in

[00:16:06] long context more

[00:16:07] in our models. Yeah.

[00:16:09] Like sort of along that line one one

[00:16:11] axis in which in which models have

[00:16:13] changed a lot and hardware has changed a

[00:16:14] lot is in like numerics right? Like I

[00:16:16] think everyone is sort of pushing

[00:16:18] towards like lower and and lower bit

[00:16:20] representations. How are you thinking

[00:16:22] about numerics? Like what what will the

[00:16:24] matics chips support and and like how

[00:16:26] are you thinking around sort of the ML

[00:16:27] consequences of your numerics scheme

[00:16:28] here? I mean numerics has been the

[00:16:31] single biggest

[00:16:32] sort of

[00:16:34] actual improvement in flops per watt of

[00:16:37] of anything like anyone has done over

[00:16:39] the last 10 years. It's it's always been

[00:16:40] that. So Nvidia has been having the

[00:16:42] precision multiple times starting at 32

[00:16:44] 16 8 4.

[00:16:46] And I think it's with good reason. The

[00:16:48] the reason it I mean empirically the

[00:16:50] results are good. If I you know going

[00:16:53] all the way from 32-bit to 8-bit like it

[00:16:55] was almost a free lunch. You you have

[00:16:57] the precision. There's zero loss in

[00:16:58] quality. Like like as long as you get

[00:17:01] the recipe right you don't even have to

[00:17:02] increase the number of parameters or

[00:17:04] anything. It's it's just like free lunch

[00:17:06] totally.

[00:17:08] We've gone past that like

[00:17:11] leadingly obvious like free lunch thing

[00:17:13] and and now it's like when you go from

[00:17:14] 8-bit to 4-bit

[00:17:16] you do lose quality but

[00:17:18] uh

[00:17:19] but not too much and so for example

[00:17:21] maybe I have to add 50% more parameters

[00:17:24] to keep the quality the same but on

[00:17:26] Nvidia I get a three times speed up

[00:17:27] going from 8-bit to 4-bit and so

[00:17:30] um

[00:17:30] uh twice the parameters three times

[00:17:32] speed up it's isn't a two times speed

[00:17:34] up. Like we we see this trend.

[00:17:37] We really want to lean into it. We

[00:17:40] One of the challenges is kind of to meet

[00:17:41] the customer where they are in I mean

[00:17:43] both the customer and the research where

[00:17:45] it is

[00:17:46] and I think there's always the challenge

[00:17:48] with hardware because the hardware

[00:17:49] timelines are so much longer than

[00:17:51] than ML research timelines.

[00:17:53] If you kind of try and ask the question

[00:17:55] of where are people designing models for

[00:17:57] today maybe the answer is FP8. That that

[00:18:00] is much too backwards looking. So so you

[00:18:02] have to be a little bit old in in in

[00:18:04] your bets on numerics but then you also

[00:18:05] have to figure out how to

[00:18:07] manage the risk that you that that your

[00:18:09] prediction is wrong. So

[00:18:13] without going into specifics some of

[00:18:14] what we do there is

[00:18:16] our ML team has done amazing research on

[00:18:17] on actually what numerics we we feel we

[00:18:19] can trust and I I think at this point we

[00:18:21] have as far as I know

[00:18:23] from our point of view of all things

[00:18:26] considered of cost and flexibility and

[00:18:31] we have a great numerics solution in our

[00:18:33] chip.

[00:18:35] Our team has like validated every sort

[00:18:38] of all of that and seen we can train

[00:18:39] train models. We know what the quality

[00:18:41] loss is there and when considering

[00:18:43] quality versus speed it's it's really

[00:18:45] attractive.

[00:18:46] Like one of the other aspects that shows

[00:18:47] up here is like because of the

[00:18:49] uncertainty of prediction like

[00:18:51] how do we manage the sort of the tail

[00:18:52] risks of like maybe

[00:18:54] maybe models will turn out worse than we

[00:18:56] thought or maybe models will turn out

[00:18:57] better than we thought in in the

[00:18:58] response to recession and so we've got

[00:19:00] some nice ideas there too.

[00:19:02] Great. And and something interesting to

[00:19:04] me is that like you're one of the few

[00:19:06] sort of newer [clears throat] chip

[00:19:07] companies that are targeting both

[00:19:08] training and inference models right?

[00:19:09] Like why why make this choice? I mean

[00:19:11] like on the surface it seems that like

[00:19:13] inference is sort of like the easy

[00:19:14] obvious choice for this both from like a

[00:19:16] software perspective and also just like

[00:19:18] from a from like thinking about the

[00:19:19] hardware.

[00:19:20] Do you want to talk a bit about like why

[00:19:21] why make this choice to do both training

[00:19:23] and inference here? Yeah so I mean like

[00:19:26] there's one angle is the historical

[00:19:28] angle which is like to say

[00:19:30] why did all of those chips in 2017

[00:19:32] choose to do inference only? And that

[00:19:34] was a genuinely simpler problem than

[00:19:35] training because no interconnect

[00:19:37] actually was a big thing. Like you could

[00:19:38] make a single chip like without any

[00:19:40] interconnect it can run like a LSTM of

[00:19:42] the day or a convolution of the day.

[00:19:45] It's like 500 million parameters fits in

[00:19:47] the memory of one chip. Like single chip

[00:19:48] solution really really easy to do.

[00:19:50] That is not the case at all today right?

[00:19:52] Like we're running inferences on

[00:19:53] hundreds of chips or more um

[00:19:55] even for inference. And so

[00:19:57] the inference chips that anyone makes

[00:19:59] today have to be rack scale systems um

[00:20:01] or bigger um

[00:20:03] and you've bitten off already most of

[00:20:05] the complexity that shows up, the system

[00:20:06] complexity for a training solution, too.

[00:20:09] So, the delta from that to a training

[00:20:11] solution is

[00:20:12] maybe for training you want scale up

[00:20:14] solution. Maybe for inference you want

[00:20:16] that, too, especially if you're doing

[00:20:17] weights and Astra and you need very

[00:20:18] large uh um connectivity domains.

[00:20:21] Um

[00:20:22] and then

[00:20:24] what what else is there in the delta? Um

[00:20:26] maybe you want a different set of

[00:20:27] precisions for training. Um sometimes

[00:20:29] training is done at higher precision.

[00:20:30] Sometimes there are other tricks that

[00:20:31] apply in training and not inference uh

[00:20:33] for for precision. Um

[00:20:35] that delta is also pretty small in

[00:20:37] reality. Um

[00:20:39] And then and then training has transpose

[00:20:41] operations, I guess is the other one.

[00:20:43] And so um

[00:20:44] but all of these things are kind of

[00:20:46] relatively small in in terms of uh

[00:20:48] the

[00:20:49] actual impact on on your chip design. Um

[00:20:52] and so

[00:20:55] I mean, you have to make a prediction

[00:20:56] about how much of the market will be

[00:20:57] training, but I think probably at least

[00:20:59] 30%. Probably not 50% because there's

[00:21:02] this RL component that is uh

[00:21:03] substantially inference, too.

[00:21:06] But for, you know, a 5% increase in your

[00:21:08] development effort, you can you can get

[00:21:10] another like

[00:21:11] uh 1/3 of the market, which is a pretty

[00:21:13] attractive place to be. So, uh I think

[00:21:15] that's the like logical calculus that

[00:21:17] was behind it. Um

[00:21:19] This is borne out by

[00:21:21] the current incumbents are all both

[00:21:23] training and inference chips in Nvidia,

[00:21:24] Google, Amazon. Um

[00:21:28] uh as well as like there's just this

[00:21:30] story of like as a customer point of

[00:21:32] view, you get this optionality of I can

[00:21:34] buy a fleet for inference, but I then

[00:21:37] get the possibility of running training

[00:21:38] on it as well. Um and so the, you know,

[00:21:42] a like especially as a customer I'm I'm

[00:21:44] thinking I'm going to be committed to

[00:21:46] these chips for 3 years 2 years I don't

[00:21:48] from now I don't know what the world's

[00:21:49] going to look like could be totally

[00:21:50] different. And so getting a bit of

[00:21:51] optionality there is uh is also like in

[00:21:53] this in this aspect is

[00:21:55] is a nice bonus feature, I think. Yeah,

[00:21:57] that makes sense. The the timeline here

[00:22:00] sort of seems pretty central to like a

[00:22:02] lot of the decisions being made, right?

[00:22:03] So, like I don't know, let's talk about

[00:22:04] timeline. Like if I wanted to start

[00:22:06] making a chip right now, right? Like I

[00:22:08] start the design process now. Like when

[00:22:10] when do I get my chip? Like when do I

[00:22:11] have it in a system in a in a data

[00:22:13] center? Yeah, so like I I have an idea

[00:22:15] for a chip and I want to do like the

[00:22:16] architecture and then the RTL to TV and

[00:22:19] uh and then tape out and and so on. Um

[00:22:22] Development times from like product idea

[00:22:25] to uh to tape out are typically several

[00:22:27] years. Um that's probably accelerating

[00:22:29] with AI. Um

[00:22:31] uh

[00:22:32] Not as much as in as in software because

[00:22:36] a big part because of risk management

[00:22:37] that um

[00:22:39] redoing a tape out I mean the tape out

[00:22:40] is not that exp- that expensive. It is

[00:22:42] only $30 even if I much more than that

[00:22:44] on model runs these days. Um but but

[00:22:47] really like the cost of a respin is not

[00:22:49] just the tape out, but then also like

[00:22:50] you've deployed billions of dollars

[00:22:52] chips in the field if you have um and so

[00:22:54] so there's there's a whole

[00:22:56] like in reality there's something on the

[00:22:58] order of a few billion dollars of like

[00:22:59] total

[00:23:00] uh project at stake there that that

[00:23:02] causes you to be still more conservative

[00:23:04] in hardware than in software.

[00:23:05] Um and I think that sort of back

[00:23:07] propagates to

[00:23:08] uh

[00:23:09] the success of uh of AI as applied to

[00:23:13] chips has not been as good to date as as

[00:23:15] applied to software. Um I'm sure that's

[00:23:16] going to like improve over time, but uh

[00:23:18] Where where where do you see this like

[00:23:20] accelerating things the most? Like is it

[00:23:21] I don't know,

[00:23:22] are the models like fairly good at

[00:23:24] writing Verilog? Is it is it something

[00:23:25] else in the stack?

[00:23:26] >> Yeah, the um I mean most of the stack is

[00:23:28] software. So, like or like is shaped

[00:23:30] like software development. So, uh so

[00:23:32] they're pretty good at writing Verilog.

[00:23:34] Um they're

[00:23:35] uh which is a lot of the development. Uh

[00:23:37] they, you know, they can also do design

[00:23:39] verification. There's this question of

[00:23:41] like, okay, they've done that. You trust

[00:23:42] it. You trust the coverage. Maybe you

[00:23:43] should like uh review it somewhat

[00:23:45] yourself as well. Um

[00:23:47] But um I think they're pretty good

[00:23:49] there. Um

[00:23:51] We haven't seen them be so good on the

[00:23:55] product and architecture side yet. Um I

[00:23:57] think that also maybe requires different

[00:23:59] workflows for how you do product and

[00:24:00] architecture. Um but like a lot of I

[00:24:03] think maybe more so in in hardware than

[00:24:06] what I've seen in software, there's a

[00:24:07] lot more um work done sort of the whole

[00:24:10] model is more waterfall than than what

[00:24:11] you see in software typically. And so

[00:24:13] there's a lot more work done um prior to

[00:24:15] writing code. So, writing architecture

[00:24:17] docs um

[00:24:19] uh modeling performance out and so on.

[00:24:20] Um

[00:24:21] The any any time where it's modeling

[00:24:23] performance, you can do that in Python.

[00:24:24] Models will do a pretty good job of

[00:24:25] that. But uh but sort of reasoning

[00:24:27] through a a lot of the other stuff, I

[00:24:29] think there's there's a component there

[00:24:30] that is

[00:24:31] more shaped like prose than than code.

[00:24:34] Um and so that's that's not

[00:24:36] to date not such a a great fit. Um And

[00:24:38] then there's also an another aspect

[00:24:41] which is not at all shaped like code.

[00:24:42] Physical design is like uh

[00:24:45] is based in a in a GUI and you're like

[00:24:48] um

[00:24:49] there's actually a lot of physical work

[00:24:50] and then slow iteration times where you

[00:24:52] do a placement run index a week or 2

[00:24:54] weeks or something like that. Um

[00:24:56] uh and so not so well adapted to the

[00:24:57] kind of flows that we have today. Um I

[00:25:00] don't know, I think all of this can get

[00:25:01] a lot faster, but like sort of if you

[00:25:03] look at what is the um

[00:25:05] limiting factor there, there is still

[00:25:06] the uh there's I don't know, a year year

[00:25:09] and a half 2 years from from tape out to

[00:25:11] production deployment, which is like

[00:25:13] entire physical supply chain and and

[00:25:14] tape out and so on.

[00:25:16] In terms of like sort of the the the

[00:25:18] tools being relatively slow and the

[00:25:19] iteration iteration times being long,

[00:25:20] right? Like at what point do you just go

[00:25:22] and, you know, rewrite the stack on the

[00:25:23] on these things, right? Like I don't

[00:25:24] know, it's become a lot easier to I mean

[00:25:26] the stack is just software, right? And

[00:25:27] so like Yeah. At what point do you just

[00:25:29] go in and say like we're just going to

[00:25:31] rewrite the stack here and and make

[00:25:32] things much faster and and like I don't

[00:25:34] know, get better iteration times. Are

[00:25:35] are people doing this already? Yeah, I

[00:25:37] think it's an interesting question. Um

[00:25:39] so, I mean, to be clear about that

[00:25:40] stack, that stack is the EDA tool stack

[00:25:43] which is produced by so EDA electronic

[00:25:45] design automation. Um

[00:25:47] It's primarily produced by Cadence and

[00:25:49] Synopsys. Um and the reason those tools

[00:25:52] are so expensive firstly is because they

[00:25:54] embody in them the um the trust me I

[00:25:58] I've done the tape outs and when I when

[00:26:00] I give you a design that I say will meet

[00:26:02] TSMC's like design rule checks and uh

[00:26:05] will will meet timing and won't uh like

[00:26:08] won't have electrical interference

[00:26:09] between all of the wires. All of these

[00:26:10] yucky physical problems that we don't

[00:26:12] want to deal with. Um

[00:26:13] like that's why we pay them the the

[00:26:15] money for that. Um uh

[00:26:17] So, uh like that is quite a different um

[00:26:21] style of work than at least what we as a

[00:26:23] fabless semiconductor company do. Um I I

[00:26:26] sure hope either like Synopsys and

[00:26:28] Cadence are are doing this uh rewriting

[00:26:30] themselves or and I'm sure there's

[00:26:31] probably a bunch of startups as well

[00:26:32] doing that as well. Um but there is um

[00:26:35] sort of some painful like decades of

[00:26:37] experience of what what are all the

[00:26:38] things that can go wrong in making sure

[00:26:40] you you um you meet all of the design

[00:26:42] rules um that uh that is maybe something

[00:26:45] of a bit of a barrier to entry or at

[00:26:47] least like um

[00:26:49] you're going to have to like waste a few

[00:26:50] hundred million dollars in in in bad

[00:26:51] tape outs before you you get that

[00:26:53] experience. Yeah.

[00:26:54] Expensive. Yeah.

[00:26:56] Yeah. And then like my impression is

[00:26:58] that uh both like allocation for wafers

[00:27:00] and then HBM are becoming increasingly

[00:27:02] scarce. Like A like I don't know, what

[00:27:04] does that look like today, right? And B

[00:27:05] like how do you design a chip um for

[00:27:08] this knowing that like I don't know,

[00:27:09] like some of the things you may have

[00:27:10] banked on are potentially hard to come

[00:27:12] by or much more expensive than you

[00:27:13] initially anticipated. Yeah, I mean, the

[00:27:16] Yeah, I don't know, it's kind of weird,

[00:27:17] right? Like you you can maybe sign a

[00:27:19] contract with one of your vendors and

[00:27:20] say, well, I, you know, will agree to

[00:27:23] logic dies at this price, but then like

[00:27:25] ultimately that might the supply might

[00:27:27] not exist. And so you've agreed to a

[00:27:28] price that like

[00:27:31] like just like the the supply and demand

[00:27:33] uh just doesn't intersect and and we'll

[00:27:34] have to renegotiate the price in in the

[00:27:36] future. So, there is that possibility

[00:27:37] and that unfortunately exists with logic

[00:27:39] dies and and memory dies as well. Um To

[00:27:42] some extent where uh I mean, what are

[00:27:43] the kinds of moves you can make in this

[00:27:45] place? Um the the most fundamental thing

[00:27:47] is just to

[00:27:48] um

[00:27:49] to have better economics there. So, uh

[00:27:52] the if you can get better performance

[00:27:55] out of every square millimeter of

[00:27:56] silicon, that means you can you can pay

[00:27:58] more than your competition for for a

[00:27:59] wafer um

[00:28:01] when you're like at the uh if if you're

[00:28:03] selling at cost, basically. So, if

[00:28:04] you're selling at cost uh um

[00:28:07] we can sustain um higher costs than than

[00:28:09] say our competition because we get more

[00:28:10] value out of the every square millimeter

[00:28:12] of silicon. Um likewise for for HBM

[00:28:14] bandwidth. Um if you can put every byte

[00:28:17] per second of HBM bandwidth to more

[00:28:19] productive use than then you can outbid

[00:28:21] your competitors while still, you know,

[00:28:23] um being price performance uh parity.

[00:28:27] Um So, that I think is the

[00:28:29] like first principles uh reasons that

[00:28:32] um

[00:28:33] we think we can have a sustaining

[00:28:35] advantage here. Um then there's the just

[00:28:38] like the day-to-day uh

[00:28:40] um business side of signing the right

[00:28:41] deals and and and forming the right

[00:28:42] partners. And I think uh

[00:28:45] I mean, without going into too much

[00:28:46] specifics, I think making sure you have

[00:28:47] the right partnerships and and uh and

[00:28:50] getting close to some of the big players

[00:28:51] is is important there.

[00:28:53] And in terms of sort of the overall

[00:28:54] supply here, like I don't know, how are

[00:28:55] things looking going forward? Like are

[00:28:57] is is HBM sort of hard to come by now or

[00:29:00] >> Yeah, no, HBM probably is. Logic wafers

[00:29:01] a little bit um

[00:29:03] uh and and then um

[00:29:05] uh advanced packaging is also hard to

[00:29:06] come by. The whole the whole ecosystem

[00:29:08] is very scarce, but uh but I think like

[00:29:10] everyone feels this across the whole

[00:29:11] supply chain. Yeah. Yeah, sort sort of

[00:29:13] going back to like the the, you know,

[00:29:15] you can if you can get more like flops

[00:29:17] or performance per square millimeter,

[00:29:19] like you can have better pricing. Like

[00:29:21] how how do you see sort of pricing and

[00:29:23] and margins going forward, right? Like I

[00:29:24] think for the last couple of years like

[00:29:26] Nvidia has been able to like sustain

[00:29:27] these insane margins because there's

[00:29:29] been very little competition and I think

[00:29:30] this is like starting to change, right?

[00:29:31] So, like how are how are you sort of

[00:29:32] reasoning about this in terms of like I

[00:29:34] don't know, like how do you price a chip

[00:29:35] in a world where there's like a couple

[00:29:36] of different chips that are potentially,

[00:29:38] you know, able to be used for training

[00:29:40] and inference effectively? Yeah, so I

[00:29:42] mean, our own pricing strategy, I mean,

[00:29:44] what

[00:29:44] we're selling to maybe five different

[00:29:47] customers in in the world. So like we we

[00:29:49] don't really need to put our price on

[00:29:50] the website. We're like we're going to

[00:29:52] we're just going to negotiate it with

[00:29:53] with with

[00:29:55] end customers.

[00:29:56] Um

[00:29:57] Uh so how would you price your strategy?

[00:29:59] Ultimately, we're going to we'll we'll

[00:30:01] negotiate it. Um when it gets to the

[00:30:03] point where we're like being asked to

[00:30:04] sell at cost, like we we have to say no,

[00:30:06] sorry. Um but uh

[00:30:08] um

[00:30:10] So like uh but like zooming out and

[00:30:12] saying um how would you expect the

[00:30:14] industry to to react in in both a like a

[00:30:17] place where there's not a lot of

[00:30:18] competition, you should be able to

[00:30:19] charge high margins. Um and then as

[00:30:21] there is more competition, the um the

[00:30:23] margins should in general come down. So

[00:30:25] I I think that will be the case over the

[00:30:26] next few years. Um

[00:30:28] The I mean, both in the current

[00:30:30] incumbents, like uh Nvidia, Google,

[00:30:32] Amazon, um

[00:30:34] uh Google being willing to sell outside

[00:30:37] of its own data centers, which we've

[00:30:38] seen some instances of. Um More more of

[00:30:42] Google's traditional competitors buying

[00:30:43] Google's chips, um rather than Nvidia's

[00:30:46] or in in conjunction with Nvidia's. Um

[00:30:49] uh Open AI buying AMD chips. All of that

[00:30:51] is like signs that even even among the

[00:30:53] traditional chips, uh margins are coming

[00:30:54] down. Um and we sort of see that with

[00:30:57] some of the uh deals where Nvidia is um

[00:31:00] is not discounting the chips on a cash

[00:31:01] basis, but maybe on a like there's some

[00:31:03] equity deal that that effectively

[00:31:05] functions as a discount. But it looks

[00:31:06] better on Yeah, the accounting

[00:31:08] department.

[00:31:08] >> Yes. Uh

[00:31:10] um so I think margins are coming down.

[00:31:12] Um

[00:31:13] The

[00:31:14] like

[00:31:15] As a startup, we we have to plan for

[00:31:18] from our perspective the worst, which is

[00:31:19] that like we had we we we sell at cost

[00:31:21] plus 1% or something like that. Um

[00:31:23] The

[00:31:25] I don't think we will sell for that low

[00:31:26] of a price. I think we'll sell it like

[00:31:28] way more than that. Um The the price

[00:31:31] that we expect to be forced to is price

[00:31:33] performance parity with with with with

[00:31:35] the best competitor. Um and we we will

[00:31:37] be very happy at that point. Right. And

[00:31:39] in that case, like performance is now is

[00:31:41] now sort of the the selling point,

[00:31:42] right? And so like And and and so like I

[00:31:44] don't know like like why why Matics,

[00:31:46] right? Like why why Matics sort of went

[00:31:48] on this on this axis versus I guess like

[00:31:50] the the potentially many competitors

[00:31:52] that'll that'll be out there. So

[00:31:53] performance. Um There's throughput and

[00:31:55] latency. Um Latency the latency bar

[00:31:58] historically has been just the latency

[00:32:00] get out of get out of HBM. It takes 20

[00:32:02] milliseconds to read all of HBM on most

[00:32:04] generations of HBM. Like it gets like

[00:32:07] that's has been remarkably stable from

[00:32:09] like HBM1 through HBM4. Um which is

[00:32:12] really just a function of you get a

[00:32:13] bandwidth increase, but you also get a

[00:32:15] capacity increase. And so

[00:32:17] uh

[00:32:17] that 20 milliseconds has essentially set

[00:32:19] what has been typically the the latency

[00:32:21] of a of a inference forward pass. And so

[00:32:24] uh

[00:32:24] you you can do speculative coding

[00:32:26] because you get a bit of an advantage

[00:32:27] over that, but uh

[00:32:28] ballpark uh HBM-based systems generally

[00:32:31] see a few hundred uh OTBS, uh which is

[00:32:33] coming from this uh this drain time of

[00:32:35] HBM.

[00:32:36] That um that table stakes is probably

[00:32:39] going to become like more like a

[00:32:41] millisecond rather than 20 milliseconds.

[00:32:42] So thousandish. One to 2,000 OTBS. Um

[00:32:46] Uh because of I think general industry

[00:32:48] industry switch over the next few years

[00:32:51] to

[00:32:52] um weights in SRAM.

[00:32:53] Um

[00:32:54] I don't see much path for going

[00:32:56] substantially below that um in the short

[00:32:58] term.

[00:32:59] Um

[00:33:01] It really depends on model sizes. Things

[00:33:02] you can do with smaller models,

[00:33:03] especially very dense models. There

[00:33:05] there are tricks that you can go uh

[00:33:06] below that. Tellus has put out a nice

[00:33:08] demo, which is like uh about an order of

[00:33:10] magnitude better than that, but that

[00:33:11] like relied on a very particular sweet

[00:33:14] spot of not a sparse model, very small

[00:33:16] model, so that you can fit in one chip

[00:33:18] the whole model. Um but once you sort of

[00:33:21] fall out of that sweet spot, you you

[00:33:23] like I think it'll generally go back to

[00:33:24] about the millisecond uh per forward

[00:33:25] pass.

[00:33:26] Um So performance uh on the latency

[00:33:29] aspect, that is to say, I think

[00:33:31] there is a new table stakes. Matics hits

[00:33:33] that. Probably a few other companies

[00:33:35] will hit it more and more well over

[00:33:36] time. Um uh I think that will sort of be

[00:33:39] end up being the new sort of table

[00:33:41] stakes and neutral point.

[00:33:44] From that neutral point, then the

[00:33:45] question is do you win on throughput?

[00:33:48] That comes down to really two things. Um

[00:33:53] What is the flops you're getting out or

[00:33:54] flops per square millimeter or flops per

[00:33:55] dollar? Um

[00:33:57] as well as what is your ability to

[00:33:59] support like long context uh well and

[00:34:02] manage the

[00:34:03] KB cache load times.

[00:34:04] So uh we'll do the flops first. Um so

[00:34:08] uh Matics has the most flops per square

[00:34:10] millimeter of any announced product um

[00:34:12] by a pretty good margin. Um

[00:34:15] This is um

[00:34:18] coming from from some of the work we've

[00:34:19] done in numerics. It's also coming from

[00:34:21] uh just some of the architectural work

[00:34:23] we've done to uh

[00:34:25] avoid a lot of the overheads of like

[00:34:27] communication inside the chip, as well

[00:34:30] as um too much flexibility that you see

[00:34:32] in other solutions. Um this is sort of

[00:34:34] what I was saying about

[00:34:36] um overheads from like supporting

[00:34:38] convolutions or low batch size things.

[00:34:40] Like if you decide not to do those

[00:34:41] things, you can you can make a

[00:34:42] substantially simpler architecture.

[00:34:45] Um

[00:34:46] All of that together ends up meaning

[00:34:47] that we have a lot better flops.

[00:34:49] Um

[00:34:51] And then on the

[00:34:52] uh KB cache side, this is really

[00:34:55] hybrid

[00:34:56] weights in SRAM, KBs in HBM, uh and

[00:34:59] really the model architecture the

[00:35:02] principle that you have in mind there is

[00:35:04] um sparse attention. So not fetching all

[00:35:06] of KBs every time, but fetching a small

[00:35:08] subset of the KBs every time. Th- That

[00:35:10] that is sort of how you make uh like

[00:35:13] achieve high throughput uh

[00:35:15] uh even though your weights are in SRAM.

[00:35:16] So uh if if you do not have this HBM

[00:35:19] available, you would be forced into a

[00:35:20] low batch size in order to fit all of

[00:35:22] the KBs. Um And a low batch size uh

[00:35:25] limits the amount of performance you can

[00:35:26] get or the amount of throughput you can

[00:35:27] get.

[00:35:29] And so like I don't know, we've kind of

[00:35:30] talked about like, you know, the the

[00:35:32] sort of hardware approaches the

[00:35:33] different chips have taken over the last

[00:35:35] couple years. There's like the SRAM-only

[00:35:36] ones, there's like the the more

[00:35:37] traditional like

[00:35:39] HBM-first ones and then sort of like

[00:35:40] what Matics is doing. Let's talk about

[00:35:42] software. Um yeah. To to me it seems

[00:35:44] like most the sort of uh chip companies

[00:35:47] in the in the last sort of 10 years have

[00:35:48] tried to build out a full stack, right?

[00:35:50] Where like I don't know, you you write

[00:35:52] some JAX or or PyTorch and and you get

[00:35:54] something running on the the model. This

[00:35:55] is not the approach you're you're trying

[00:35:57] to take here. Yeah.

[00:35:58] >> Um Let's talk about why.

[00:36:00] So what are the successes of this

[00:36:02] approach? Um I think Google has been the

[00:36:04] clear success. Uh Google had had a new

[00:36:07] hardware that was not a GPU um and they

[00:36:10] um made JAX work really well on it. Um

[00:36:14] uh and so at least like take that as a

[00:36:16] case study in my own experiences of like

[00:36:18] being at Google and writing JAX code,

[00:36:20] it's pretty productive. You can get to

[00:36:21] maybe good like 50% MFU or so uh writing

[00:36:24] sort of straight-up JAX without like at

[00:36:26] the high level, JAX XLA.

[00:36:29] Uh 50% is pretty good performance, but

[00:36:31] like you might hope for 70%, 90% if you

[00:36:33] can.

[00:36:34] To get that last mile, um it seems to

[00:36:37] always be the case in practice that you

[00:36:39] you're going to want to write custom

[00:36:40] kernels.

[00:36:41] Um it's just like I think it's a reality

[00:36:43] that like if you try and teach a

[00:36:45] compiler how to do good optimizations,

[00:36:47] it is so much work because you have to

[00:36:48] think, well, a compiler could

[00:36:51] get any kind of input at all. And so the

[00:36:53] specific optimization that I know how to

[00:36:55] do in this specific case,

[00:36:56] now I have to generalize and say, well,

[00:36:58] I could get a thousand different cases

[00:37:00] and I have to figure out how to do the

[00:37:01] optimization in all of these cases. So

[00:37:03] just as a sort of

[00:37:05] um an engineering problem of

[00:37:07] am I more productive solving the

[00:37:08] optimization in in one case versus a

[00:37:10] thousand cases? Like obviously in one

[00:37:11] case it's it's an easier problem. So

[00:37:14] So that incentive pushes you towards

[00:37:16] wanting to write custom kernels for for

[00:37:18] most of your model over time.

[00:37:20] Um and from what I can see, that is uh

[00:37:24] that is the reality of how it has borne

[00:37:26] out in the labs. So um the labs uh have

[00:37:28] large uh people teams of people writing

[00:37:31] custom kernels for the platforms they're

[00:37:32] they're on. Um and

[00:37:34] uh and I think for ultimately for the

[00:37:36] for this reason, um it is worth spending

[00:37:38] the human time to uh to make the model

[00:37:40] cheaper. And uh and you can get pretty

[00:37:43] big wins there.

[00:37:44] Um

[00:37:45] So from our perspective, let's lean into

[00:37:47] that. Uh if you know that your customers

[00:37:49] are going to be writing custom kernels,

[00:37:59] for 10%

[00:38:00] That's also nice for us because it says,

[00:38:02] okay, we're not going to have to spend a

[00:38:03] huge amount of time developing a

[00:38:04] compiler that uh that's not going to be

[00:38:06] so useful.

[00:38:07] Um I think all of this is predicated on

[00:38:11] sophisticated users. Um This strategy

[00:38:14] would not work at all for um if we're

[00:38:17] trying to sell to a small shop that has

[00:38:18] only like 10 engineers or something like

[00:38:20] that because they they need they need

[00:38:22] the productivity there more than more

[00:38:23] than the the the performance. Yeah. I

[00:38:26] guess the one place where to me it it

[00:38:27] seems that this like approach runs into

[00:38:29] some headwinds is like when you're doing

[00:38:30] sort of like research and like like

[00:38:32] training research um type workloads,

[00:38:34] right? Like what what are your thoughts

[00:38:35] on that? Like is this something that

[00:38:36] you're just sort of not targeting for

[00:38:38] now or like is there sort of a plan in

[00:38:41] the future to to address that sort of

[00:38:42] use case where like flexibility is is is

[00:38:44] useful, right? And like um writing you

[00:38:47] know, the entire model as kernels is is

[00:38:49] potentially hard. Yeah. Um so I mean, I

[00:38:51] think the simplest answer is for now

[00:38:53] that's not a main focus for us. Um

[00:38:55] uh

[00:38:56] like we are not trying to say to a lab,

[00:38:58] you will we will be 100% of your chips.

[00:39:00] We're happy to be

[00:39:01] 50%, 75%. That would be a lot of

[00:39:04] business to have. Happy to be 10% as

[00:39:05] well. I I think that leaves a lot of

[00:39:07] room to say uh you do your um research

[00:39:09] on on on a platform that's optimized for

[00:39:12] um flexibility and productivity. Um and

[00:39:15] then when you're doing uh switching to

[00:39:17] like uh hero runs or like starting to

[00:39:19] scale up your model to prepare for a

[00:39:20] hero run, or then when you've trained

[00:39:22] your model and you want to deploy it, uh

[00:39:23] that's where you switch to a different

[00:39:25] um

[00:39:26] uh to a different platform like Matics.

[00:39:28] Um maybe as a slightly hot take, I kind

[00:39:31] of

[00:39:32] like

[00:39:34] I believe this is not universally the

[00:39:35] case, but I think um this is a

[00:39:37] reasonable approach to have even if

[00:39:38] you're on one platform, in fact. That uh

[00:39:41] maybe you should just have a separate

[00:39:42] code base for uh research and

[00:39:43] production. Have them as two separate

[00:39:44] code bases

[00:39:46] even if both of them are going to be on

[00:39:48] on GPUs or Google or something like

[00:39:49] that. Um just from the point of view of

[00:39:52] like

[00:39:53] I'm optimizing for a very different set

[00:39:55] of constraints here than than for here.

[00:39:57] And like why would I force them to be

[00:39:58] the same thing?

[00:39:59] I might force them to be the same thing

[00:40:00] so I can do a transfer like a transfer

[00:40:02] from research to production really

[00:40:03] effectively. Um that is a valid goal to

[00:40:06] have. But other goals you might also

[00:40:08] have are can I have my researchers be

[00:40:09] extremely productive and not by the same

[00:40:13] like same set of constraints which is

[00:40:15] that I want production to be like very

[00:40:16] high

[00:40:17] and and

[00:40:19] bandwidth utilization as well. So like

[00:40:21] don't necessarily don't over couple

[00:40:24] together the two different use cases you

[00:40:25] have. Um

[00:40:26] which I think it plays in software but

[00:40:28] like more strongly it also plays in

[00:40:29] hardware. Right. It's also harder to

[00:40:30] switch back and forth on hardware for

[00:40:32] that, right?

[00:40:33] Yeah, that that that that makes sense to

[00:40:35] me. Um

[00:40:36] And then I guess like what what is this

[00:40:37] like like let's say I want to write my

[00:40:39] model on on an Amatic's chip, right?

[00:40:41] Like what what does this look like? Is

[00:40:42] it something It's obviously not

[00:40:43] something like JAX but like is it

[00:40:44] something like Triton? Is it something

[00:40:46] lower level? Are you going to give me

[00:40:47] your ISA? Yeah. Yeah. Yeah. Um

[00:40:50] We'll we'll give you our ISA wrapped in

[00:40:52] a Python DSL. Um Uh I think that's sort

[00:40:56] of it gives

[00:40:57] you as a customer gives you what you

[00:40:58] want from the point of view of I want to

[00:41:00] know the instructions that are running

[00:41:01] so I can debug if the performance is not

[00:41:03] as good as I want. I understand why. I

[00:41:05] expect that I mean in our own workflows

[00:41:07] we look at

[00:41:08] at the instruction level of what are the

[00:41:10] ALUs I have? What is the mapping of my

[00:41:12] instruction to those ALUs? Are there

[00:41:13] like resource bubbles in here? And I

[00:41:16] and you know people writing kernels are

[00:41:18] are going to be doing the same thing. So

[00:41:19] exposing the ISA pretty closely is is

[00:41:22] um

[00:41:22] uh it's just like lets you use it where

[00:41:25] they want. I think like

[00:41:27] that is the lowest level of what we do.

[00:41:29] Sort of aspirations for future I think

[00:41:31] the aspiration for future is more of a

[00:41:34] Triton or Palace like thing. That is a

[00:41:36] pretty nice level of abstraction I think

[00:41:39] the teams that at

[00:41:40] at OpenAI and Google did a really good

[00:41:42] job there

[00:41:43] of

[00:41:45] like you don't want something too high

[00:41:46] level. The risk of going too high level

[00:41:47] is that you're having the computer make

[00:41:49] decisions like the compiler make

[00:41:50] decisions that

[00:41:51] you could have made better yourself. Um

[00:41:54] and so

[00:41:55] like don't hand over the keys to the

[00:41:56] kingdom.

[00:41:57] Um But at the same time you do want the

[00:41:59] compiler to make decisions that are not

[00:42:00] very interesting for you to make and a

[00:42:01] computer can make better. And so I think

[00:42:03] uh Triton and Palace did a pretty good

[00:42:05] job with that. The The The user is

[00:42:07] making decisions on

[00:42:09] definitely HBM transfers

[00:42:11] transfers over interconnect

[00:42:14] and even like loop order which are like

[00:42:16] those are the most important decisions

[00:42:17] to make. You should totally make them by

[00:42:18] hand. Um and then but then they had it

[00:42:20] over to the compiler the sort of

[00:42:22] short-term decisions of SRAM loads and

[00:42:24] stores and and sort of inner loop order.

[00:42:27] Um I think

[00:42:30] so that theme of like picking where to

[00:42:31] make those decisions and go a little

[00:42:32] high level is nice.

[00:42:34] Um

[00:42:35] To some extent it is still a little too

[00:42:37] high level because

[00:42:39] it's appropriate for chips like TPUs and

[00:42:42] GPUs where

[00:42:43] um SRAM bandwidth is abundant or L1

[00:42:46] cache bandwidth is abundant

[00:42:48] but that's actually leaving some

[00:42:49] performance on the table or silicon area

[00:42:51] on the table. And so

[00:42:53] if you want to recover that performance

[00:42:55] as well then it sort of pushes you to

[00:42:57] maybe a slightly lower level of

[00:42:59] abstraction still. Mhm. Yeah.

[00:43:01] We've we've talked about TPUs in in

[00:43:02] Google a lot. Um You've spent a lot of

[00:43:05] time at Google. Yeah. What what did you

[00:43:07] do there? What what what did you you

[00:43:08] know? Yeah. So I um

[00:43:11] So I was at Google for about a decade. I

[00:43:14] I started

[00:43:15] as a as a web developer

[00:43:17] which is just like in 2012 what you

[00:43:19] would do. Um

[00:43:21] I love it. Did you make the Amatic's

[00:43:22] website?

[00:43:23] I I made an early version of the

[00:43:24] Amatic's website. When it when it looked

[00:43:26] really ugly I made it. Like

[00:43:27] my making it

[00:43:29] things look nice was never my area

[00:43:31] really my area but one of the things I

[00:43:32] cared about a lot with the Amatic's

[00:43:33] website is like as a company we make

[00:43:36] things really efficient and we should

[00:43:37] have the same attention to detail with

[00:43:38] with the website.

[00:43:40] I you know the the original version of

[00:43:42] the Amatic's website was about 5 or 10

[00:43:43] kilobytes. Like it loads really fast. No

[00:43:45] no no like no redundant requests. All

[00:43:48] all loads in one request. Loads very

[00:43:49] fast. So yeah and I put some work into

[00:43:51] something. I I It's it's maybe a little

[00:43:53] obsessive but it like it's it's the kind

[00:43:55] of thing I like to do.

[00:43:57] So I yeah I spent like maybe 5 years

[00:43:59] doing that at Google.

[00:44:00] And then

[00:44:01] and then I I wanted to switch into some

[00:44:04] of the machine learning teams and so I

[00:44:05] did

[00:44:06] Originally I did some software work on

[00:44:08] large scale logistic regression which is

[00:44:11] like a one layer neural network but that

[00:44:14] was before

[00:44:15] the the neural network revolution.

[00:44:17] Um and then as that revolution started

[00:44:19] to happen I I moved over to a

[00:44:22] to a neural net chip team. So this was

[00:44:25] actually a competitor to the TPUs. Um

[00:44:28] There's some sort of a little bit of

[00:44:30] shared lineage. Jonathan Ross from Grok

[00:44:33] was on that project for a very short

[00:44:34] amount of time before he went to start

[00:44:35] Grok. Um

[00:44:38] and then and then most recently after

[00:44:40] working on chips I moved to the LM team.

[00:44:43] So I was on Brain.

[00:44:45] Uh

[00:44:46] I helped train Google Palm

[00:44:48] and then I wrote the inference stack for

[00:44:49] Palm as well. And so that was

[00:44:51] like very very hands-on with LLMs there.

[00:44:54] Yeah.

[00:44:55] And and I guess like now at Amatic's,

[00:44:57] right? Like I mean I'm sure you know you

[00:44:59] have a lot of sort of non-technical work

[00:45:02] that you have to get to but like when

[00:45:03] when you do have sort of when you do a

[00:45:04] technical work like what do you spend

[00:45:06] your time on? Like what Yeah, like what

[00:45:07] are what are you spending your time on?

[00:45:09] Currently the places that I'm closest to

[00:45:10] are the the ML research and and the

[00:45:12] architecture. Um

[00:45:14] Uh in the very early days of Amatic's I

[00:45:16] was also pretty deep in the

[00:45:18] in some of the software stack but since

[00:45:20] then it's been rewritten and much better

[00:45:22] by by people who have more attention and

[00:45:24] and and experience than I do on that.

[00:45:26] But I like I think it's so important for

[00:45:28] a hardware company to be very close to

[00:45:30] the ML research as in both to directly

[00:45:32] inform our product our entire numeric

[00:45:35] stack was was

[00:45:36] a result of our ML research.

[00:45:38] But then also to uh have as good of a

[00:45:41] crystal ball as we can for where models

[00:45:42] are going. Um

[00:45:44] and so

[00:45:45] I think I think the work that the team

[00:45:46] does there is really cool and exciting.

[00:45:48] On and then on the architecture side

[00:45:51] I think architecture is sort of the

[00:45:52] heart obviously of a company like this.

[00:45:53] And so

[00:45:54] there's like that ranges from the like

[00:45:57] the sort of coarse-grained thing of like

[00:46:00] how do you organize the chip? How do you

[00:46:01] connect all of the cores of the chip

[00:46:02] together?

[00:46:03] Lots of interesting things you can do

[00:46:04] there. But then it goes down into like

[00:46:06] even to the microarchitecture level like

[00:46:09] one of the examples is like um

[00:46:13] the kinds of co-design you can do where

[00:46:14] you say well um there are these

[00:46:17] standards for how you do rounding. Round

[00:46:18] to nearest ties go to to the nearest

[00:46:20] even is like one of the IEEE standards.

[00:46:23] Um but it's really overkill.

[00:46:25] And so if you like simultaneously pay

[00:46:27] attention to the microarchitecture of

[00:46:28] what does a circuit for that rounding

[00:46:30] look like? Um and then at the same time

[00:46:33] can I prove with my ML

[00:46:35] modeling

[00:46:36] work that I could do a cheaper circuit.

[00:46:39] Like the the marriage of the

[00:46:40] microarchitecture and and and the ML

[00:46:41] model I think is a really

[00:46:44] sort of attractive opportunity for for

[00:46:45] co-design. Mhm. Yeah. On the on the sort

[00:46:48] of co-design front, right? Like

[00:46:50] I mean some some competitors, right?

[00:46:52] Like I don't know OpenAI is making their

[00:46:53] own chip. And and and so like

[00:46:56] as as sort of external to a to a lab and

[00:46:58] like obviously Google, you know, does

[00:47:00] both their own hardware and and their

[00:47:01] own models. And so like how are how are

[00:47:03] you sort of trying to co-design for

[00:47:05] something that you, you know, don't know

[00:47:07] the exact shape of and and still sort of

[00:47:09] beat out your competitors on that? Yeah.

[00:47:10] And and I think one aspect of this is

[00:47:12] the more general thing of like is

[00:47:15] are the advantages of vertical

[00:47:16] integration? How So

[00:47:18] there are advantages of vertical

[00:47:19] integration. You get more information

[00:47:20] about what your workload is.

[00:47:22] And then compare that to the advantages

[00:47:24] of having like

[00:47:25] multiple customer customers chip

[00:47:27] customers buying from a single chip

[00:47:29] vendor. There are advantages to

[00:47:30] centralization as well. And so like

[00:47:32] long-term there's a question of which

[00:47:33] one which of those paradigms wins.

[00:47:36] Um

[00:47:37] the

[00:47:38] So just to spell that out historically

[00:47:40] Nvidia has won. Uh like the um you know,

[00:47:44] they are selling to multiple different

[00:47:45] buyers and so they can aggregate the

[00:47:47] research dollars into into one product.

[00:47:49] Whereas when there are multiple

[00:47:50] different buyers those research dollars

[00:47:52] have to be spent sort of duplicately

[00:47:54] and you can't do as much research. I

[00:47:56] think that is that is a reason to to

[00:47:58] think that actually a a standalone

[00:48:00] company like Amatic's could succeed

[00:48:03] even despite if for example OpenAI

[00:48:05] developing their own chips. Um

[00:48:07] In the short term how do we do co-design

[00:48:10] um when when we're not when we don't

[00:48:12] have this information? We have to do the

[00:48:13] best job we can of getting information

[00:48:14] of what models are.

[00:48:16] Um

[00:48:17] It was easier in 2022 when people

[00:48:19] published all the models.

[00:48:21] They stopped doing that in '22. Maybe

[00:48:23] OpenAI and Anthropic stopped that maybe

[00:48:25] in '21 a little bit earlier. But

[00:48:28] but at this point everyone has stopped

[00:48:29] except for the the Chinese labs.

[00:48:32] So

[00:48:33] uh

[00:48:34] Open source as a source of information

[00:48:35] is still reasonably good especially

[00:48:37] coming from DeepMind. They've uh

[00:48:40] put a lot of really informative papers

[00:48:42] out. To some extent you can talk to

[00:48:43] customers. You can't get a lot there

[00:48:45] because customers really value their own

[00:48:46] IP.

[00:48:48] But you can you can get a little bit.

[00:48:49] And then you have to do your best job of

[00:48:51] covering the distribution of likely

[00:48:53] outcomes. So like maybe the 80%

[00:48:56] central part of the distribution and

[00:48:58] ignore the 10% 20% tails.

[00:49:00] Um

[00:49:01] So

[00:49:02] that is sort of what we do. We like a

[00:49:04] way to think about it is we look at the

[00:49:05] primitives that are used in neural nets.

[00:49:07] So obviously matrix multiplication uh

[00:49:10] standards like floating point vector

[00:49:13] operations to do all the common set of

[00:49:15] non-linear functions.

[00:49:16] And then there's some things that you

[00:49:17] need for sparsity and then maybe some

[00:49:20] different routing modes for

[00:49:22] inference and training. Um

[00:49:24] There is that set of primitives. That

[00:49:26] set of primitives as a class has been

[00:49:27] extremely stable over the last 5 10

[00:49:29] years in fact of neural nets. And so

[00:49:31] betting on the primitives uh is actually

[00:49:34] a pretty safe place to be. And I think

[00:49:36] it's very unlikely that

[00:49:38] model changes a year or two years from

[00:49:40] now are going to move outside of that

[00:49:41] set of primitives. It's just because

[00:49:43] like it's kind of the intersection of

[00:49:45] what what is efficient on hardware or in

[00:49:47] physics, what is efficient in physics,

[00:49:49] as well as what is efficient for

[00:49:52] gradient descent. And so so

[00:49:54] like that is a starting place that gives

[00:49:56] us sort of a um

[00:49:58] a

[00:49:59] foundation of stability.

[00:50:01] Kinds of things we do for for actual

[00:50:03] co-design is or or like using knowledge

[00:50:05] about models is um what is the what is

[00:50:08] the ratio of those resources, how much

[00:50:10] we put in.

[00:50:11] Um

[00:50:12] And and that becomes a hard question

[00:50:14] when it's

[00:50:15] when it's a costly decision of how much

[00:50:17] of this resource versus that resource.

[00:50:19] So how much interconnect is is a pretty

[00:50:21] costly decision because it's quite

[00:50:23] expensive. In other places it's not so

[00:50:25] costly, like how much

[00:50:27] sorting throughput is actually a pretty

[00:50:29] like it's pretty cheap to provision, so

[00:50:30] you can we can over provision that.

[00:50:32] Um so that's sort of our methodology in

[00:50:34] general. Um

[00:50:36] the

[00:50:39] Where this ends up is that

[00:50:42] uh

[00:50:43] the information advantage or

[00:50:45] disadvantage um like we think we have a

[00:50:48] pretty good crystal ball. Is is

[00:50:49] ultimately

[00:50:50] I think that that is a place of

[00:50:52] differentiation for us. We think we have

[00:50:53] a better crystal ball than than many. Um

[00:50:56] and how can that even be the case when

[00:50:57] you're comparing to a lab that has their

[00:50:59] own like internal folks? Um

[00:51:03] To put it bluntly, it's uh

[00:51:05] the even when you talk to the

[00:51:06] researchers they don't know what is

[00:51:07] going to be the case in three years. And

[00:51:08] so like you can talk to the researchers

[00:51:10] all you want, they can tell you what

[00:51:11] they're doing today, but if you're

[00:51:12] trying to predict two or three years out

[00:51:13] you still need to make your own

[00:51:14] prediction. And so we think we can do a

[00:51:16] good job with that. Mhm. And this is

[00:51:18] flowing from sort of the the numerics

[00:51:19] and and ML work you're doing while while

[00:51:21] designing this chip. Yeah, as well as

[00:51:23] just like um like having seen some of

[00:51:25] the like longer term history of of of of

[00:51:27] of the development of neural nets over

[00:51:29] time. Mhm. Are [clears throat] you

[00:51:30] you're you're the crystal ball here? Um

[00:51:33] Uh

[00:51:34] to a large extent, yeah.

[00:51:34] >> [laughter]

[00:51:35] >> Um I mean

[00:51:36] uh

[00:51:37] we

[00:51:39] There's a lot of folks on the team. We

[00:51:40] talk a lot to the ML team. Um and then

[00:51:42] the architecture team is also

[00:51:43] gets quite close to the ML research as

[00:51:45] well. And so I think um

[00:51:47] like considering all of those things in

[00:51:49] mind, uh that's that we do the best job

[00:51:51] we can there. We've sort of touched on

[00:51:53] this, I think we just did in this past

[00:51:54] conversation, but like let's let's talk

[00:51:55] a bit about interconnect. Um like what

[00:51:58] does your interconnect look like? Like

[00:51:59] how is it different from what's out

[00:52:00] there right now? And like like how does

[00:52:02] this play into sort of the the

[00:52:03] trade-offs you're making here and then

[00:52:05] in terms of just sort of maximizing your

[00:52:06] flops and and making things as simple as

[00:52:08] possible.

[00:52:10] Yeah, um

[00:52:12] so where is interconnect

[00:52:14] where is it expensive, what is it used

[00:52:16] for? So um

[00:52:18] uh you need enough interconnect.

[00:52:19] Primarily primarily it's driven by

[00:52:22] you need enough interconnect to get over

[00:52:24] a certain bar of something. Um

[00:52:28] That bar is actually different for

[00:52:29] HBM-based chips and SRAM-based chips. Um

[00:52:31] the traditional bar on HBM-based chips

[00:52:33] is I want to do enough um internal

[00:52:36] parallelism within a layer, so tensor

[00:52:38] parallelism or expert parallelism. I

[00:52:40] need to do enough of that in order to

[00:52:41] get my latency down. Um at just to some

[00:52:44] amount. Um and so like you then look how

[00:52:47] much of that I need to do, how much

[00:52:49] compute throughput do I have. Um

[00:52:52] uh and then how much HBM bandwidth do I

[00:52:54] have and and form form a decision there.

[00:52:57] The bar for SRAM is actually phrased

[00:52:59] slightly differently. It is a

[00:53:02] I want to stick my weights in SRAM and

[00:53:04] then uh I need to be able to like so

[00:53:06] that means I can only do a certain

[00:53:07] amount of computation before I run out

[00:53:09] of SRAM and need to move to the next

[00:53:10] chip. And so I need enough interconnect

[00:53:12] that in that short amount of time I can

[00:53:14] get the the the activations in and out.

[00:53:16] Um

[00:53:17] uh

[00:53:19] But that actually does give give you a

[00:53:20] relatively clear threshold for how much

[00:53:22] interconnect you want. You have to

[00:53:23] couple that with some predictions about

[00:53:25] neural architecture um in terms of

[00:53:29] um it's actually not too sensitive to

[00:53:31] the size of a model, but it's it is

[00:53:32] sensitive to some of the other hyper

[00:53:33] parameters of a model. Um so there is

[00:53:35] some amount of prediction that goes on

[00:53:36] there.

[00:53:37] What we have done on interconnect is

[00:53:40] um provisioned a lot of it, for sure. Um

[00:53:42] the the logic die cost of that is

[00:53:45] actually um

[00:53:47] not too bad. Uh I mean it is one of the

[00:53:49] big ticket items like big ticket items

[00:53:51] on a logic die are interconnect, SRAM,

[00:53:53] and compute. Um and so it's it is up

[00:53:55] there, but it's not it's not a it

[00:53:56] doesn't balloon out of control. The

[00:53:58] other cost is like the cables and the

[00:53:59] power cost. And again it is a big ticket

[00:54:01] item there, but it's not it's not out of

[00:54:03] control. Um

[00:54:05] the So so we have provisioned a lot

[00:54:07] there. One of the things that is sort of

[00:54:09] different is that um

[00:54:12] a side benefit and we're going into too

[00:54:14] much details, a side benefit of

[00:54:16] weights in SRAM is actually it allows

[00:54:18] you to um use your interconnect without

[00:54:22] also using up HBM bandwidth, which is a

[00:54:24] shocking thing, but

[00:54:25] we're able to do that. And so like push

[00:54:27] the interconnect performance higher than

[00:54:29] than for example some of the incumbents

[00:54:31] are able to um for that reason. So

[00:54:34] that is not really talking about any of

[00:54:35] the numbers or the topology of what we

[00:54:37] do, but

[00:54:38] we have a lot of interconnect. Um

[00:54:40] uh

[00:54:41] We've picked a what we think is a pretty

[00:54:43] good topology as well. Um it is

[00:54:45] different than some of the previous uh

[00:54:48] topologies.

[00:54:49] Um

[00:54:50] Uh I think this is a point of innovation

[00:54:51] for us. The kinds of considerations when

[00:54:53] you uh pick an interconnect topology are

[00:54:56] firstly just what are the main

[00:54:57] operations like collectives you're going

[00:54:58] to do on there. Um

[00:55:01] how do you keep like interconnect

[00:55:03] latency down? Um and then how do you

[00:55:06] scale up really well? And so we're in

[00:55:08] the same sort of family of what the all

[00:55:11] of the players are doing. We don't have

[00:55:12] like um

[00:55:13] like totally different off-the-wall

[00:55:15] ideas, but we uh the way we've we've

[00:55:17] tuned it is I think pretty nice. Mhm.

[00:55:19] He's just like so

[00:55:21] Wow.

[00:55:24] Okay, so like you have this chip that

[00:55:25] doesn't exist yet, right? And you like

[00:55:27] want to run some workload on it and the

[00:55:28] workload also doesn't exist yet. So like

[00:55:30] obviously modeling is is pretty

[00:55:31] important. Um how do you think about

[00:55:34] this, right? Like like Do you have any

[00:55:35] takes on on how performance modeling

[00:55:37] should look and and and what you're

[00:55:38] doing? Yeah, yeah. I think uh I mean I

[00:55:41] think at baseline let's just like

[00:55:44] to set expectations like performance

[00:55:46] modeling is so much more important in um

[00:55:48] AI workloads than just the how people

[00:55:51] tune and optimize AI workloads than what

[00:55:53] we've seen before. Like if you contrast

[00:55:55] with like the work of a someone doing

[00:55:58] performance engineering on a CPU versus

[00:56:00] on a GPU or like AI workload. Um

[00:56:05] And when we're doing uh AI workloads, we

[00:56:08] reason about percentage percentage of

[00:56:09] peak performance. And we think we can

[00:56:10] get like 70%, 90%, 50%, something in

[00:56:13] that range.

[00:56:14] Percentage of peak performance on a CPU,

[00:56:16] what does that even mean? Like are you

[00:56:18] ever going to be able to utilize all the

[00:56:19] flops or like percent like can you

[00:56:22] utilize all of the branch predictor

[00:56:23] performance? Like how how can you even

[00:56:25] reason about that? And so like as a

[00:56:27] starting point, the first thing you do

[00:56:29] when optimizing like an LLM is you sit

[00:56:32] down and say what do I expect the

[00:56:34] performance to be?

[00:56:35] One of the things that uh actually I

[00:56:38] remember like first hearing this from

[00:56:40] Gaurav Agrawal, who was um

[00:56:43] at Google and now at OpenAI, um is like

[00:56:45] what is what is your goal when you're

[00:56:46] doing performance modeling? Um like you

[00:56:48] might think my goal is to make a

[00:56:49] performance estimate that is as precise

[00:56:51] as possible.

[00:56:53] But there's no end to that, right?

[00:56:54] Because like as precise as possible

[00:56:55] means I start off modeling the

[00:56:56] coarse-grained things like memory

[00:56:58] bandwidth and compute performance. And

[00:56:59] then I go finer grained and I look at um

[00:57:03] uh maybe how the instructions are

[00:57:04] scheduled against my resources. And then

[00:57:06] maybe I even have to I could be even

[00:57:08] more precise than that and say well,

[00:57:10] what's the power modeling? Am I am I

[00:57:11] going to get thermally throttled?

[00:57:13] There's no end to it, right? Like you

[00:57:14] just keep going. Um

[00:57:17] So where do you stop with performance

[00:57:19] modeling is is sort of a interesting

[00:57:21] meta-question. And so the question is

[00:57:23] like maybe I'm trying to get

[00:57:25] good enough to answer a question that

[00:57:27] I'm I'm trying to answer.

[00:57:28] So there's a few different takes on

[00:57:30] that, but I I think like there's this

[00:57:31] perennial theme of how can I approximate

[00:57:33] to answer the question faster?

[00:57:36] Which I don't know. At least for for me

[00:57:38] like learning this in school I hated it,

[00:57:40] right? Like we we went to like spend a

[00:57:41] lot of time in physics class class and

[00:57:43] they kept saying you should approximate.

[00:57:44] And like it's crazy. Like you're like

[00:57:46] I've got this question, I can solve it

[00:57:47] exactly, and they're saying you should

[00:57:49] approximate it rather than solving it

[00:57:50] exactly. It like

[00:57:52] uh it defies some kind of moral

[00:57:54] aesthetic that you have about like why

[00:57:55] are you throwing stuff out? Um

[00:57:58] But I I at least for me I found like

[00:58:00] this really turned around when I'm

[00:58:02] actually trying to answer questions

[00:58:03] quickly and like even in my head. And

[00:58:06] approximation is totally necessary

[00:58:08] there. So one of the exercises that like

[00:58:10] to go through this is like how can you

[00:58:12] figure out which approximations to make?

[00:58:13] How are you comfortable decide like

[00:58:15] making these approximations but not

[00:58:16] those approximations? And what is the

[00:58:18] sort of principled rationale behind

[00:58:20] that?

[00:58:21] Um we have some like mathematicians on

[00:58:25] our team with a very strong mathematical

[00:58:26] background and they're like

[00:58:27] approximations like

[00:58:29] I want to prove something, how do I do

[00:58:30] that when I'm approximating? Um

[00:58:32] So one theme here is

[00:58:35] you're not trying to prove the

[00:58:36] performance is exactly this. That's too

[00:58:38] hard to prove. You'll never finish. Um

[00:58:41] and so maybe you're trying to prove

[00:58:42] something. If you want to view it

[00:58:43] through the lens of what am I trying to

[00:58:44] prove, you would like to bound the

[00:58:46] performance. So performance cannot

[00:58:48] possibly be a bit better than this.

[00:58:50] Roofline analysis is like the kind of

[00:58:52] canonical example of that.

[00:58:53] Um

[00:58:54] uh but you can actually like do that

[00:58:56] kind of analysis in in lots of contexts.

[00:58:59] Um sometimes you prove the opposite,

[00:59:00] which is like the performance will be at

[00:59:01] least this good. That's that's a harder

[00:59:03] thing to do, but

[00:59:04] um or maybe you do asymptotic

[00:59:06] performance, but that's not so like um

[00:59:09] it throws out too much information.

[00:59:11] Like doesn't distinguish between 30%

[00:59:12] AMFU and 90% AMFU.

[00:59:14] Um

[00:59:15] But uh

[00:59:16] a theme of can I prove an upper bound um

[00:59:19] I think is sort of the strongest

[00:59:21] like provable and how to go about

[00:59:23] things. Now is that upper bound useful?

[00:59:25] Like if I have an upper bound saying the

[00:59:26] performance will be better than 100%,

[00:59:28] that's not very useful. Um and then I

[00:59:29] measure it and I find actually it's 1%,

[00:59:31] and so well, it is within that upper

[00:59:32] bound, but it's totally useless. Um so,

[00:59:35] then you have to think about like what's

[00:59:36] the tightness of my my bound. So, like

[00:59:39] it can't be better than 50%. I've got

[00:59:42] 30% in reality. That's actually

[00:59:43] reasonably tight. Um

[00:59:45] what is the tightness gap? Now,

[00:59:47] uh

[00:59:48] and there I say there's a tightness gap

[00:59:50] that's maybe coming because of when I

[00:59:52] look at my profile when I'm actually

[00:59:53] running it, um there was this what is

[00:59:56] the single biggest effects that I didn't

[00:59:58] model there and maybe if I include that

[01:00:00] in my modeling now, I can say my bound

[01:00:01] is not 50% but it's 35%. Uh and so, now

[01:00:04] it becomes increasingly tight. That is a

[01:00:06] sort of

[01:00:07] like a

[01:00:08] s- uh

[01:00:09] a mathematical basis that I think has

[01:00:10] sound principles to to reason about. And

[01:00:12] I think sort of like that's in in the

[01:00:14] back of uh my mind always when when

[01:00:16] doing performance modeling. I would say

[01:00:18] like we don't really always formalize it

[01:00:20] cuz it's a bit too laborious like

[01:00:22] proving theorems every single time, but

[01:00:23] but that's the kind of proof you would

[01:00:24] look for.

[01:00:26] Um

[01:00:27] I think

[01:00:28] like curiously, if you think about what

[01:00:30] theorems can I prove in other contexts,

[01:00:32] there's kind of a branch and bound thing

[01:00:34] going on where like I've got a search

[01:00:36] space over chips that I want and I say,

[01:00:37] well, I could consider chips that have

[01:00:38] this certain characteristic. Can I

[01:00:40] immediately bound the performance of

[01:00:42] anything in that class and say,

[01:00:44] no matter what I do, if I make this one

[01:00:46] design decision, I can't possibly get

[01:00:47] good performance. This is like the HBM

[01:00:49] SRAM thing. Yeah, for example. Um and

[01:00:51] so,

[01:00:52] uh

[01:00:53] yeah, exactly. So, like there's actually

[01:00:54] a proof you can do that there that you

[01:00:56] look at like the relationship between

[01:00:58] the number of uh interconnect hops, um

[01:01:01] the amount of capacity, you can look at

[01:01:03] the capacity and connect it to the batch

[01:01:04] size via Little's Law, um and you can

[01:01:07] sort of prove bounds on the uh

[01:01:09] utilizable flops um there.

[01:01:11] Or um or maybe I've got a system where

[01:01:14] I've got two two chips inter-

[01:01:17] interacting with like completely

[01:01:18] different resource profiles, um

[01:01:20] can I prove a bound on like

[01:01:23] uh any way I connect them, this chip is

[01:01:25] going to limit the performance of that

[01:01:26] chip or something like that. Um

[01:01:28] And so, uh I think sort of the exercise

[01:01:30] always when trying to prove it is like

[01:01:32] um prove the bound and then like

[01:01:36] actually like prove the upper bound on

[01:01:37] performance can't be better than this

[01:01:38] and then um actually work something out

[01:01:41] in detail and say, well, I can achieve

[01:01:42] this and then and then look at the

[01:01:44] tightness there. Um so, yeah, HBM SRAM

[01:01:46] is is one of the strongest places we've

[01:01:48] done that um and like I I think that

[01:01:50] from first principles that sort of just

[01:01:52] says SRAM for these kinds of workloads

[01:01:53] can't um

[01:01:55] can't be attractive uh SRAM alone. And

[01:01:58] you can then also examine your

[01:01:59] assumptions about about model

[01:02:00] architecture like a model has at least

[01:02:01] 100 layers or has at least a certain

[01:02:04] amount of bytes of KB cache per per

[01:02:05] token or something like that. Um and

[01:02:07] sort of like just do the exercise of

[01:02:09] exploring way we do that.

[01:02:10] Mhm. And and just to give some clarity

[01:02:12] to this, like what does this exercise

[01:02:13] look like in practice, right? Like how

[01:02:14] are you like like is is is there some

[01:02:16] like big spreadsheet that you're like

[01:02:18] plugging numbers into or are you like

[01:02:19] writing out proofs here? Like what does

[01:02:21] this look like, right? And then like how

[01:02:22] how does this play into the process of

[01:02:24] designing the chip, right? Like is it

[01:02:25] like you do a couple of rounds of

[01:02:26] iteration, you do some like simulation

[01:02:28] here? Um what does this sort of

[01:02:29] end-to-end process look like to

[01:02:32] you know, kind of arrive at like what

[01:02:33] configuration in this space you actually

[01:02:35] want to go with? So, we we have big

[01:02:37] spreadsheets a lot. Um and those

[01:02:38] normally like uh

[01:02:40] those are normally saying what we

[01:02:42] believe is achievable or we can

[01:02:43] demonstrate in very high fidelity, we

[01:02:45] can at least achieve this if not better.

[01:02:46] Um

[01:02:47] big spreadsheets, Python modeling, um

[01:02:50] uh like writing out instruction traces,

[01:02:52] all of those things are ways of saying

[01:02:54] this is achievable. Um the the upper

[01:02:56] bounds actually we try and um minimize

[01:02:59] the the assumptions of the proof as much

[01:03:01] as possible. And so, those end up being

[01:03:02] quite short actually. So, um

[01:03:05] like it's sort of like the can you see

[01:03:06] it at a glance? Like can I prove an

[01:03:08] upper bound that relates only like three

[01:03:09] variables together or five variables

[01:03:10] together um to the point where I can do

[01:03:12] that in my head. Um uh so, we we do that

[01:03:15] exploration a few iterations and I mean

[01:03:17] a lot of this is stuff that we would do

[01:03:18] very early in a chip project um and that

[01:03:22] tends to uncover um

[01:03:24] like your goal is to like eliminate as

[01:03:26] many variables from the from the bounds

[01:03:28] as you can to like make it a stronger uh

[01:03:30] result, especially if you can eliminate

[01:03:32] actually hyper parameters about the

[01:03:33] model architecture because like you

[01:03:34] don't want to say like this chip works

[01:03:36] for models that have like a D model

[01:03:38] dimension of 8,000, but if it goes to

[01:03:40] 16,000, then it's not going to work

[01:03:41] because like that's that's way too

[01:03:43] specific. Um and so, uh

[01:03:46] like to take tensor parallelism as an

[01:03:48] example, um you tend to find and this is

[01:03:50] like there's like actually the scaling

[01:03:53] book on on performance estimation for um

[01:03:55] tensor parallelism and and expert

[01:03:57] parallelism talks about this. You tend

[01:03:59] to find and this is just how it

[01:04:01] turns out is that um

[01:04:04] you you get results of like how big of a

[01:04:06] matrix per chip rather than how big of a

[01:04:07] matrix in total. Then you can read like

[01:04:09] that's actually sort of nicer to reason

[01:04:10] about in hardware and say, well, okay,

[01:04:12] well, then that actually tells me how

[01:04:13] big my SRAM should be. Mhm. So, a lot of

[01:04:16] those calculations early on in

[01:04:18] coarse-grained sizing of HBM SRAM

[01:04:19] interconnect flops. Um

[01:04:22] as it proceeds, um we do the same kind

[01:04:25] of methodology but into the

[01:04:27] finer-grained details like um what

[01:04:29] should the ratio of my um

[01:04:32] uh sorting performance versus my matrix

[01:04:34] performance or my um like softmax

[01:04:37] performance to matrix performance or

[01:04:38] something like that be. And and there

[01:04:39] again like if you're trying to minimize

[01:04:41] it, it it's maybe connected to what is

[01:04:42] the um minimum efficient contraction

[01:04:45] size that I'm going to support in my

[01:04:46] chip. Um

[01:04:47] or uh

[01:04:49] uh sort of what level of sparsity do I

[01:04:51] support is connected to the amount of uh

[01:04:53] feature these resources. Right. And and

[01:04:55] sort of like how long is this design

[01:04:57] process, right? Where you're sort of

[01:04:58] like going back and forth between

[01:05:00] um whatever your like sort of latest

[01:05:01] state is on on on the design and then

[01:05:03] and then sort of going back to like

[01:05:05] modeling things out again. And also like

[01:05:06] I don't know, like in in this whole sort

[01:05:08] of design process, right? Like what is

[01:05:09] the longest step, right? Like what takes

[01:05:11] the most time? Like um is is it like the

[01:05:14] verification? Is it just like I don't

[01:05:15] know, like writing out the code and then

[01:05:16] like doing this doing this like modeling

[01:05:18] exercise and like coming back and

[01:05:20] looking at your assumptions?

[01:05:21] Yeah, um I mean it it depends on how you

[01:05:23] want to run um such a project. There is

[01:05:25] sort of the minimum viable product angle

[01:05:27] on on a chip design and and then there's

[01:05:29] also the um sort of

[01:05:31] sort of uh rich design um philosophy.

[01:05:34] And we we as a company have leaned more

[01:05:35] in the latter style um which is I would

[01:05:38] say somewhat against Silicon Valley um

[01:05:41] norms which is like you you start a

[01:05:42] startup, you ship a chip uh very quickly

[01:05:43] and iterate. Um but like reflecting the

[01:05:46] reality of this market, um

[01:05:48] chip tape outs are very expensive

[01:05:50] especially when you include all the

[01:05:51] downstream deployment costs um and we're

[01:05:53] also competing against like uh TPU V7

[01:05:55] and uh like very uh mature uh GPU

[01:05:58] products. Um we have chosen to to do

[01:06:01] quite a like a mature design which is

[01:06:03] sophisticated in many different axes uh

[01:06:05] simultaneously. That has led to I mean

[01:06:07] combination of that and like like we're

[01:06:10] substantially innovating on the way we

[01:06:12] um do our numerics and uh connect uh all

[01:06:15] all of the cores of the chip together

[01:06:16] and so on. We've spent a a reasonable

[01:06:18] amount of time on the um

[01:06:20] uh actually early stage architecture and

[01:06:22] really tuning that and and mapping that

[01:06:23] to uh what information we have about

[01:06:25] models and what we see as that changes.

[01:06:27] Um but then downstream of that there is

[01:06:29] like the actual implementation of the

[01:06:30] chip is actually quite a long process as

[01:06:31] well. So, uh

[01:06:32] 1 to 2 years is is is common for for

[01:06:34] many places. I I guess like on the on

[01:06:36] the topic of modeling, right? Like I

[01:06:38] mean part of the question here is like

[01:06:39] what are you modeling, right? It's like

[01:06:40] you you have some some like LLM in mind,

[01:06:43] right? And and I I think though like the

[01:06:45] the workloads tend to differ like differ

[01:06:47] a lot between like prefill and decode,

[01:06:48] right? So, like how how do you reason

[01:06:50] about that divide? Um

[01:06:51] I think like you know, like I think this

[01:06:53] also sort of plays into like

[01:06:55] fundamentally what your chip is designed

[01:06:57] for and like what the tradeoffs you're

[01:06:58] making here, right? Like how are you

[01:06:59] trading off between these two things?

[01:07:01] Yeah, so I mean I would say

[01:07:04] uh

[01:07:04] like

[01:07:05] so, how are prefill and decode

[01:07:07] different? Prefill is very demanding on

[01:07:09] compute performance, has relatively

[01:07:10] little demands on on memory bandwidth.

[01:07:12] Um decode is the opposite there. Um

[01:07:15] and yet we run them on the same chip. Uh

[01:07:17] everyone runs it on the same chip. Maybe

[01:07:18] there's some area for specialization of

[01:07:20] of chips and video has done a little bit

[01:07:21] of that, but they're basically still the

[01:07:23] same chip with very small modifications.

[01:07:26] So,

[01:07:27] we're constrained that we want both of

[01:07:29] them to run well on the same hardware

[01:07:31] and yet the resource footprints are the

[01:07:33] same. That can't possibly go well. Like

[01:07:35] something's something's wrong there. And

[01:07:37] then like even more than that, not even

[01:07:39] considering the hardware, but um

[01:07:42] everything about how we train a model uh

[01:07:44] is oriented towards supporting decode.

[01:07:45] That means I can incrementally add new

[01:07:47] tokens. Um there's this causal mask.

[01:07:50] Causal mask prevents things like um

[01:07:52] uh bidirectional attention. Um there are

[01:07:54] so many places that the constraints of

[01:07:56] decode show up um in how I train a model

[01:07:59] and they're just totally irrelevant for

[01:08:00] for prefill. And so,

[01:08:02] uh

[01:08:04] like I would say in some sense these are

[01:08:05] like I I I I see that and I hear like

[01:08:08] warning sign like something's off there.

[01:08:10] Like we're we're applying constraints

[01:08:12] that uh are artificial because of some

[01:08:14] coupling between the prefill and decode

[01:08:16] model.

[01:08:17] So, can we eliminate that coupling? I

[01:08:19] think is it is an interesting question.

[01:08:21] So, where did that coupling come from?

[01:08:22] It came from well, firstly, I train a

[01:08:24] model and I use that for both prefill

[01:08:25] and decode. Um and then uh what that

[01:08:29] means is that like the decode needs to

[01:08:30] be able to attend to the pre- to the

[01:08:31] prefill. There's a historical answer to

[01:08:33] that which is encoder-decoder models.

[01:08:35] We've heard about them like T5 is that,

[01:08:37] uh BERT is an encoder-only model. Um

[01:08:40] the early translation models are

[01:08:41] encoder-decoder coder models. Um they

[01:08:44] avoid this coupling and the encoder

[01:08:45] model can can look quite different than

[01:08:47] the decoder model.

[01:08:49] The main difference historically being

[01:08:50] that it has bidirectional attention.

[01:08:52] There are various technical reasons why

[01:08:53] you can't actually apply that in in the

[01:08:56] uh chatbot context. Um the biggest

[01:08:58] problem is

[01:08:59] so, the it's it's great. You you do you

[01:09:01] run the encoder on the user's first

[01:09:02] response.

[01:09:04] And then everything after that has to be

[01:09:05] decode because uh like you

[01:09:08] enc- encoder models are fundamentally

[01:09:09] not incremental at all. And so,

[01:09:12] uh like

[01:09:13] it's kind of a waste of effort, right?

[01:09:14] Like 1% of the conversation I'm going to

[01:09:16] run the encoder on and then 99% I'm

[01:09:18] going going to run the decoder on.

[01:09:19] That's what really that's what led to

[01:09:21] decoder-only only models uh being in the

[01:09:23] space. But there's there's There's huge

[01:09:25] gap between like encoder models which

[01:09:27] get all the tokens at once and then

[01:09:28] decoder models which get tokens one at a

[01:09:30] time. Um is there something in between

[01:09:32] where we can have like a model

[01:09:34] uh

[01:09:35] class which gets like a batch of tokens

[01:09:37] at a time? Like instead of

[01:09:40] all the tokens, maybe 100 tokens at a

[01:09:41] time or 500 tokens at a time or

[01:09:42] something like that.

[01:09:44] This totally works, at least

[01:09:45] conceptually. Um and so you can sort of

[01:09:47] split your model into two halves. The

[01:09:48] first half gets tokens a batch at a

[01:09:50] time. The second half gets tokens one at

[01:09:52] a time.

[01:09:53] Um you call the first batch your prefill

[01:09:54] model. Call the second batch your model.

[01:09:56] Uh the decoder model can cross attend to

[01:09:58] the prefill model um in the same way

[01:10:00] that encoder-decoder does cross

[01:10:01] attention.

[01:10:02] Um this uh like you can construct the

[01:10:04] causal masks um such that you're not

[01:10:06] violating causality anywhere here. Uh

[01:10:08] it's trainable. It gives you all the

[01:10:09] advantages of decoder-only training in

[01:10:10] that uh you can you you can actually

[01:10:12] have a loss function on every single

[01:10:14] token in in the sequence.

[01:10:15] Um

[01:10:17] So so this is something that we've

[01:10:18] explored a little bit. Um it's this I

[01:10:20] mean this is sort of like it's a bit of

[01:10:22] an out there idea and and it hasn't been

[01:10:23] our main area of research at all. Um but

[01:10:26] the at least the idea of if you can lift

[01:10:28] some of these constraints,

[01:10:30] uh maybe uh

[01:10:32] maybe it just gives you new

[01:10:33] opportunities that you can you can try

[01:10:34] out. So um there are so many

[01:10:38] ideas, especially in like the vision

[01:10:39] transformer scenario. Vision

[01:10:41] transformers are all uh like there's no

[01:10:43] causal masking there. They um they're

[01:10:45] they're actually in the encoder family

[01:10:46] rather than the decoder family. And

[01:10:48] there's so many interesting ideas from

[01:10:50] that literature that

[01:10:51] do not apply to decoder-only models

[01:10:53] because of the causal mask.

[01:10:54] So uh

[01:10:56] like expert choice attention doesn't

[01:10:58] work. It breaks It breaks the causal

[01:10:59] mask. Um soft mixture experts or smooth

[01:11:02] mixture experts, which is like a uh

[01:11:04] even like more extreme than than expert

[01:11:05] choice, uh also can apply in in contexts

[01:11:08] like this. Um things where you

[01:11:10] downsample uh tokens over time. So like

[01:11:12] funnel transformer uh where you say

[01:11:15] actually I'm going to have a

[01:11:15] representation over multiple tokens and

[01:11:17] then combine them together. Also breaks

[01:11:18] causal mask, but you can apply it in a

[01:11:20] in a prefill model. So I think it's an

[01:11:22] interesting research direction. Um it

[01:11:24] seems like there's so many things that

[01:11:25] like have been thrown out because of

[01:11:27] constraints of decoder models that could

[01:11:29] apply there. Um

[01:11:30] uh it's just like it's a nice thing to

[01:11:32] to explore.

[01:11:33] Yeah, something that stood out to me

[01:11:34] like throughout this whole conversation,

[01:11:35] right? Is like we've talked about

[01:11:36] everything from like model architecture

[01:11:38] all the way down to like I don't know

[01:11:39] like microarchitecture on the chip,

[01:11:40] right? And like it seems like building

[01:11:43] out this chip project requires basically

[01:11:45] like expertise at every level of the

[01:11:46] stack. And like I'm kind of curious like

[01:11:48] how how do you think about like hiring

[01:11:50] people, right? Like in building out the

[01:11:52] team and like scaling as as sort of like

[01:11:54] MatX grows as a company? Yeah. I mean I

[01:11:56] I think this is actually just what is

[01:11:57] really exciting about being in a

[01:11:58] hardware team. Like uh I think in a

[01:12:01] software or ML team, like your focus is

[01:12:03] um like up and down the stack is less

[01:12:05] than uh in a hardware company. We have

[01:12:08] people all the way from like ML models

[01:12:10] through uh compiler, people are writing

[01:12:13] um like constraint solvers in the

[01:12:14] compiler. We have uh people writing

[01:12:15] kernels like optimizing things like

[01:12:17] crazy. All of that stuff similar to what

[01:12:18] you would get in a in a lab. But then as

[01:12:20] we go down the stack, we um we're we're

[01:12:23] doing microarchitecture. What is

[01:12:25] actually the circuit for a floating

[01:12:27] point adder? Um

[01:12:29] and then even down below that, physical

[01:12:30] design. And then even just things like

[01:12:32] in in the rack, like what's the

[01:12:33] insertion force of a of a of a tray into

[01:12:36] a rack? Um what is the

[01:12:39] uh utility power draw? How do we like

[01:12:40] avoid big DIDT swings at at at the at

[01:12:43] the data center level and so on. And so

[01:12:45] it's like I think it's I mean personally

[01:12:47] this is why I'm I'm doing hardware.

[01:12:49] There's much more of a clean slate and

[01:12:51] range of things you consider. It's It's

[01:12:52] kind of a richer space to explore.

[01:12:55] Um

[01:12:56] And so like that means in hiring side,

[01:12:58] we uh we hire people with all of those

[01:13:00] different expertises. So our company is

[01:13:03] currently about 100 people. We have um

[01:13:05] I'd say about 2/3 of that is hardware um

[01:13:07] covering all of the things I said like

[01:13:09] uh logic design, physical design,

[01:13:10] architecture, design verification, uh

[01:13:13] rack system, um

[01:13:15] uh signal integrity, power integrity,

[01:13:17] all of those things. Um and then we also

[01:13:19] have um really strong software team

[01:13:22] doing uh kernels, um compilers, uh

[01:13:25] simulators, and so on. Um and then and

[01:13:27] uh and a great ML team doing all of this

[01:13:29] kind of ML research that we're talking

[01:13:30] about. Um

[01:13:32] we uh

[01:13:35] A a very successful company here needs

[01:13:37] to grow a lot. Um I think uh like in all

[01:13:39] of these places we we we we'd love to

[01:13:40] have more people and and uh yeah, we

[01:13:43] would be very happy to. What are what

[01:13:44] are you most excited about with MatX

[01:13:46] over the next year? Yeah, we got a lot

[01:13:48] of work ahead of us and I think it's

[01:13:49] going to be pretty fun. Um

[01:13:51] uh

[01:13:52] ramping volume on our first generation

[01:13:53] chip is going to be um like we the kinds

[01:13:56] of volumes we hear people talk about,

[01:13:57] really large. Um working on future

[01:13:59] generation chips as well I think is

[01:14:00] really exciting. Uh there's always this

[01:14:02] like puzzle of how how do you uh

[01:14:05] uh leave as little on the table while

[01:14:06] while still making the device work

[01:14:07] really well. Uh so like I think tons of

[01:14:10] really interesting problems to look at.

[01:14:11] That's that's incredibly exciting.

[01:14:13] Looking forward to seeing what happens.

[01:14:14] That we do.

[01:14:15] Cool. Great.

[01:14:16] Yeah, this is fun. Yeah.
