[00:00:01] Reiner Pope is the co-founder
and CEO of MatX.

[00:00:03] He's a former math whiz and Haskell
programmer who became

[00:00:06] a TPU architect for Google.

[00:00:07] Now he's teamed up with Google's former
chief chip architect to design

[00:00:10] a better chip for AI.

[00:00:16] A year ago, everyone was saying
Google is canceled.

[00:00:21] AI is going to eat their search.

[00:00:22] No one's going to search for things,
and therefore, the business won't do well.

[00:00:26] Obviously, that sentiment has really
shifted, in part helped by

[00:00:30] Gemini 3 is really good.

[00:00:32] Then also it's really fast.

[00:00:34] It's powered by the custom
chip hardware Google has.

[00:00:37] You were inside Google for a lot
of the foundational period,

[00:00:42] laying the groundwork for that stuff.

[00:00:44] What do people not
appreciate about what Google did right

[00:00:49] to lay all the groundwork
for their current AI success?

[00:00:52] They started with the research.

[00:00:54] The transformers came from there.

[00:00:55] Pretty much anyone who's
over 30 and at a large lab has

[00:00:59] been at Google Brain at some point.

[00:01:02] I think there was and has
been a lot of talent there.

[00:01:06] TPUs are pretty good.

[00:01:11] We think there's better you can do,
of course, but they at least had

[00:01:14] the option, the opportunity to design
the TPUs for neural nets at least, rather

[00:01:19] than graphics applications like NVIDIA.

[00:01:23] The overall architecture,
starting with single core doing

[00:01:27] what was at the time reasonably large
systolic arrays, by today's standards,

[00:01:30] nowhere near as much, but I think those
were a lot of really good decisions.

[00:01:35] When did the TPU project start?

[00:01:37] TPU v1 was announced in 2016, I think.

[00:01:41] That was what led to the creation
of all of those 2016–2017 startups.

[00:01:46] Cerebras, Groq, Graphcore,
SambaNova, all of those.

[00:01:50] TPU v1 is a really impressive project.

[00:01:53] It was done on a very short timeline.

[00:01:56] I don't know the full details,
but maybe about a year or so,

[00:01:58] maybe a year and a half,
with a skeleton team of 20-30 people.

[00:02:03] Really, really minimal viable product.

[00:02:06] More recent TPUs and more recent AI chips
in general

[00:02:10] can't do that because the market has
moved, and the table

[00:02:15] stakes are much higher.

[00:02:17] The first-generation product, they just…

[00:02:20] One big systolic array,
stick a memory next to it, we're done.

[00:02:24] It was really simple,
nice, elegant product.

[00:02:26] Obviously, TPU v1
predates the Transformer.

[00:02:29] Is that just a coincidence that they
happened at very similar times, or

[00:02:35] are they related in some way?
There was a period of maybe about four

[00:02:39] years of a lot of ML research, or neural

[00:02:46] net research, prior to Transformer.

[00:02:49] What was popular?

[00:02:50] LSTMs and ConvNets
and ResNet and Inception.

[00:02:54] The big thinking at the time was
to adapt it to be used for LSTMs.

[00:03:02] It's a reasonable fit there.

[00:03:04] But I think there was just
a huge flurry of activity.

[00:03:09] Why did it all happen then and not later?

[00:03:10] It's probably just because
people stopped publishing.

[00:03:13] 2022 was about the time when Google
completely stopped

[00:03:17] publishing its research.

[00:03:20] All the good papers are
from before that as a result.

[00:03:22] But is there some hand-wavy story you can
tell about parallelization where both

[00:03:26] Transformers and TPUs are

[00:03:31] about really internalizing

[00:03:34] the importance of parallelization?

[00:03:37] Definitely,
I put it somewhat on people, actually.

[00:03:42] It is just true.
Hardware is massively parallel.

[00:03:45] You've got tens of billions,
hundreds of billions of transistors

[00:03:49] on your chip, and it takes
maybe 100 clock cycles to get from one

[00:03:53] side of the chip to the other.

[00:03:55] You can't do a sequential computation
involving transistors

[00:03:58] on both sides of the chip.

[00:03:59] The hardware is just fundamentally
parallel, and you have to

[00:04:02] take advantage of that.

[00:04:03] TPU v1 and all later TPUs
naturally took advantage of that.

[00:04:08] Matrix Multiply is really nice
because it is so parallel.

[00:04:12] I think on the hardware side
that's generally understood.

[00:04:15] I think most ML researchers,
especially of the time,

[00:04:21] were not super deep in what
hardware wants, and what is…

[00:04:25] Mechanical sympathy is sometimes
a term that's used for that.

[00:04:29] What is the term?
It kind of— It speaks for itself.

[00:04:34] Think about the poor machine
and what does it want?

[00:04:37] What does it want?

[00:04:39] The term actually, I think,
originates in maybe high-frequency trading

[00:04:44] in areas like that, which is,
I haven't worked in.

[00:04:46] I've just,
I like reading about the software

[00:04:48] that people have built from there.

[00:04:51] It's like, for them,
what does the machine want?

[00:04:52] It wants a lot
of instruction-level parallelism.

[00:04:54] This is CPUs, not TPUs.

[00:04:56] Once a lot don't branch, so unpredictable

[00:05:02] branches kill your performance.

[00:05:04] Think about the things that CPUs
do and how to use them best.

[00:05:07] Can I get to peak performance on a CPU?

[00:05:09] It's that idea.

[00:05:10] I think the whole idea of peak
performance on a CPU is kind of crazy.

[00:05:15] No one even says,
"What is peak performance?

[00:05:17] What is my percentage of peak
on a CPU?" Because the performance

[00:05:21] of software running on CPUs is really bad,
but running on GPUs, or TPUs,

[00:05:25] or AI chips in general,
actually that is the main focus.

[00:05:27] It's like, "What is my percentage of peak?

[00:05:28] Can I get 70% or 80%?"
I feel like many people listening to this

[00:05:33] know that GPUs perform better
for AI workloads than CPUs.

[00:05:38] It's a funny history when you think about
it, where just

[00:05:42] one day we woke up with all these very
mathematically intensive workloads,

[00:05:47] first crypto mining, and then AI,
so then NVIDIA is extremely

[00:05:52] well-positioned because they've been
making GPUs for gamers

[00:05:56] that you would plug into...

[00:05:57] You'd buy your
Dell PC back in the day and maybe upgrade

[00:06:02] the graphics card by plugging in a better
NVIDIA graphics card than the one

[00:06:07] the stock Dell computer came with.

[00:06:10] They were incredibly
well-positioned to capture that.

[00:06:13] I think people know that.

[00:06:14] What is the intuitive explanation
as to why GPUs are better

[00:06:19] for AI workloads than CPUs?

[00:06:22] People say, "They're better for these
mathematical computations." But that's

[00:06:26] a tautological answer, basically.

[00:06:28] Is there some way you can have a mental
model for why that is the case?

[00:06:31] Because software instruction
sets also involve doing math.

[00:06:38] Intuitions, I'm not sure.

[00:06:39] Let me try and just go to
some of the big differences, which is,

[00:06:45] really wide vector instructions is
the hallmark of a GPU,

[00:06:51] which I think it's maybe…

[00:06:54] If you want some intuition, it's like how
much is spent on controlling the thing?

[00:06:59] Maybe control means,
if I'm driving a truck, how much…

[00:07:02] Is the driver versus the payload?

[00:07:04] A truck has a huge payload in it.

[00:07:06] That's more like the GPU, whereas
maybe a motorcycle is more

[00:07:09] like the CPU, where you've got…

[00:07:11] Actually just processing the instructions,
reading, "What do I have to do next?

[00:07:16] How do I do that?"
That is most of the cost on a CPU, whereas

[00:07:21] if you just keep the same instructions
but make the payload 100 times bigger,

[00:07:24] then you can shift most of the cost to be
in the actual work that you want to do.

[00:07:31] CPUs have been optimized for very complex
instruction sets, whereas

[00:07:37] GPU is optimized for—
Complex instruction sets and

[00:07:42] fine-grained changing what you want to do.

[00:07:46] Like steering, in this analogy,
a CPU can steer an obstacle course,

[00:07:49] no problem, whereas,
on a GPU, you're just going to go

[00:07:52] straight line for a really long time.

[00:07:55] This is getting us into...

[00:07:58] What is MatX?

[00:07:59] How did you guys start it,
and which part of this

[00:08:01] space are you attacking?

[00:08:02] MatX is making the best chips
physically possible for LLMs.

[00:08:13] What led us into MatX…

[00:08:16] Mike is the other founder.

[00:08:18] Mike and I were both working at Google.

[00:08:22] I was working on the inference stack
for running LLMs, and I was saying,

[00:08:28] "How can we make the best software on TPUs
for running LLMs?" Then what we really

[00:08:34] wanted out of hardware was support much,
much larger matrices.

[00:08:37] The matrices have grown from maybe 128
in dimension into the many thousands.

[00:08:41] Truck goes to many trailers.

[00:08:48] Much larger matrices and
much lower precision arithmetic.

[00:08:55] We tried to move the TPUs
in this direction.

[00:08:58] TPUs have been moving in this direction,
but they're constrained

[00:09:00] by a lot of other workloads.

[00:09:03] There was a big ads workload at the time.

[00:09:06] Back in '22, before ChatGPT was released,
there was this idea that LLMs were going

[00:09:12] to be a big thing, but not conviction,
and really hard to make a big bet on that.

[00:09:19] I think a startup is more of the right
place to make a big bet on a workload.

[00:09:23] If you fail, it's fine.

[00:09:25] Another startup will succeed.

[00:09:26] Whereas I think a company like Google or
NVIDIA, the next chip

[00:09:31] has to work for sure.

[00:09:34] You can take more technical
risks as they turn up.

[00:09:37] Actually, I would say we were taking
product risks rather than technical risks.

[00:09:41] But is there actually product risk?

[00:09:42] Because it seems like
LLMs are going to work.

[00:09:45] I think now we understand it.

[00:09:47] Two or three years ago,
I think it was just like a— And when

[00:09:51] you say the best chips for LLMs...

[00:09:52] I can think of multiple
ways to measure best.

[00:09:54] It could be the best performance per watt.

[00:09:56] It could be the lowest latency,
capable of handling the largest models.

[00:09:59] What is best?

[00:09:59] In general, there are two metrics which
LLM workloads care about, which is

[00:10:05] throughput, which is really
just an economics thing.

[00:10:08] I buy a chip for $30,000,
and then can I do 10,000 tokens a second

[00:10:13] or 100,000 tokens per
second of throughput?

[00:10:15] That determines the dollars per token.

[00:10:18] Throughput and then latency,
how fast does a thing respond?

[00:10:21] As I see the market, the economics
seems to be most important.

[00:10:28] Ultimately,
the quality of the AI you can train

[00:10:31] and serve is constrained by,
"I have only a $10 billion budget,

[00:10:35] and I want to train and serve the best
model I can on that budget." If I can have

[00:10:41] more tokens per dollar,
then I can get a better quality out.

[00:10:47] The product we aim to build is far ahead
on throughput,

[00:10:52] but then, actually,
the surprising thing is we're competitive

[00:10:56] with the best on latency as well.

[00:10:58] I think that is a unique thing
in offering both in the same place.

[00:11:01] Is this for…

[00:11:01] Obviously in AI,
there's training the models and then

[00:11:03] running the models' inference.

[00:11:05] Is this most interesting for inference,
or is there any training angle?

[00:11:09] Incidentally, is it useful for trading,
but you are trying to win inference,

[00:11:14] is that how you think about it?

[00:11:15] I think that's a reasonable
way to look at it.

[00:11:17] I think the best inference chip today will
be a really good training chip as well.

[00:11:24] Our product is both training
and inference, but I think the first

[00:11:27] sales will be an inference.

[00:11:28] That's mostly just a market effect where
it's easier to buy,

[00:11:33] it's not as big of a risk to go to buy an
inference cluster as a training cluster.

[00:11:38] I think the product is really
compelling for training as well.

[00:11:40] I think it should be
the best training product.

[00:11:43] You guys just raised a big
new round of financing.

[00:11:46] That's right.

[00:11:49] We've raised a series B round, it's led
by Jane Street and Situational Awareness.

[00:11:56] Situational Awareness, that is,
Leopold Aschenbrenner's fund.

[00:11:59] He wrote the definitive book on
AGI and where it's going.

[00:12:04] Then Jane Street, they're
real technical experts.

[00:12:07] They understand all
the details really well.

[00:12:09] Very happy to be having
them lead the round.

[00:12:13] It's a $500 million round, helps us
actually ramp the manufacturing and supply

[00:12:18] chain for our chip so we can
bring our chip to market.

[00:12:21] That's a lot of money.
It is.

[00:12:24] I think
roughly, I would say it costs a ballpark

[00:12:28] $100 million to produce
a chip in small volumes.

[00:12:32] But then you see the orders that are going
around, like OpenAI, Anthropic, Google.

[00:12:37] They're going around buying
multi-gigawatt clusters, they cost

[00:12:41] tens of billions of dollars of chips,
and you want to deploy all

[00:12:44] of that in a year or so.

[00:12:46] You just need a massive
supply chain behind you.

[00:12:50] Assuming everything works technically,
what rate of production

[00:12:55] could you start to see?

[00:12:57] We have some estimates of where
we would like to be on this.

[00:13:02] Ramping to very large volumes
is a huge challenge for anyone.

[00:13:06] Obviously, for the large players,
they've had some practice at it.

[00:13:08] Getting to a very large
volume for a startup is hard.

[00:13:12] We would like to be at a place where
we're shipping multiple gigawatts a year.

[00:13:16] Multiple gigawatts per year?

[00:13:18] Speaking of the metrics,
you talked about tokens per second.

[00:13:18] Yeah.

[00:13:21] We used to measure chips in FLOPS,
and I guess there's some kind

[00:13:24] of custom FLOP thing for AI chips.

[00:13:26] But is everyone just using
tokens per second these days?

[00:13:28] Is the industry aligning
on that as the chip metric?

[00:13:32] I guess it's like an application
metric versus the chip itself.

[00:13:37] FLOPS of the chip is the key chip metric.

[00:13:42] There's a little bit of,
if I go and say,

[00:13:44] "I've got an exaFLOP chip," to you then
the appropriate suspicion is to say,

[00:13:50] "But can I actually use those FLOPS
effectively?" Then you need

[00:13:54] to map the application to that.

[00:13:56] This is kind of telling you the usable
FLOPS for your purposes.

[00:14:00] As a consumer of AI,
we have known for a long time

[00:14:05] that lower latency products succeed.

[00:14:10] Google talked about their internal testing
where the differences were down to…

[00:14:17] Was it 50 milliseconds?
Something like that.

[00:14:19] In result times where they noticed more
Google engagement, the faster the results

[00:14:23] were, and you'd think that 50
milliseconds is imperceptible to a human.

[00:14:27] It almost is, but turns out it's not.

[00:14:30] I think Amazon has,
certainly, they've optimized the latency

[00:14:33] of the Amazon experience quite a lot.

[00:14:35] I don't know if they've talked about this
stuff publicly, but you know that their

[00:14:39] internal metrics similarly show
that the faster the product page loads,

[00:14:42] the more people buy it.
Yet in AI, Google has carved out a

[00:14:48] meaningful advantage via Gemini just being

[00:14:53] really fast for its level of intelligence.

[00:14:57] As far as I can tell,
ahead of most of the other labs

[00:15:01] at a latency, at a fixed
high level of intelligence.

[00:15:06] Why have you guys or Groq or better chips
not been adopted faster

[00:15:12] to give this product latency?

[00:15:14] It's just that this will happen and you
guys will be powering all the AI products,

[00:15:18] but I note that Google has
an interesting lead there.

[00:15:22] I think there's ultimately…

[00:15:24] At least for existing chips in the market,
there's a really uncomfortable trade-off

[00:15:27] between latency and throughput.

[00:15:29] The chips that are best at throughput have
historically been the chips

[00:15:32] that are based on HBM as the memory.

[00:15:35] That is Google, Amazon, NVIDIA.

[00:15:39] In order to have very large throughput,
you need a lot of inferences

[00:15:44] in flight simultaneously.

[00:15:45] That needs the large memory.

[00:15:47] But that hasn't been so good at latency.

[00:15:48] Then, there's the Groq and Cerebras
that are much better at latency because

[00:15:51] they've got this, the SRAM,
weights are in SRAM, very low latency.

[00:15:56] The problem is, and the challenge
when you go to a Groq or Cerebras system

[00:15:59] is that the throughput you get there,
it just is not very good.

[00:16:04] The fundamental dollars per token is just
not competitive with Google

[00:16:09] or NVIDIA or Amazon.

[00:16:11] It is actually possible
to do both in the same chip.

[00:16:15] It's kind of an obvious thing.

[00:16:16] You say you take the HBM,
you take the SRAM,

[00:16:17] put them together on the same chip,
you put the weights in SRAM, and you put

[00:16:21] all of the inference data in HBM.

[00:16:24] That is what we are doing, in fact.

[00:16:26] I think that actually hits a really nice
sweet spot where you can get low

[00:16:29] latency and also be very cheap.

[00:16:32] I think that's a really
attractive point to be.

[00:16:33] It hasn't happened in the market yet,
just because of product decisions that

[00:16:36] have been made by the different chips.

[00:16:38] But we should expect
all the AIs we're using to get

[00:16:43] significantly faster over
the coming 3–5 years?

[00:16:45] Order of magnitude faster, I'd say.

[00:16:47] Generally, HBM-based chips tend to be
about 10 milliseconds or 20 milliseconds

[00:16:52] per— I'm sorry, HBM-based
chips are things like TPUs?

[00:16:55] That's right.

[00:16:57] There's just some simple math of,
how long does it take you

[00:17:00] to read through all of HBM?
It takes about 20 milliseconds.

[00:17:02] That's the amount of time per token it
runs, whereas the amount of time to read

[00:17:06] through all of SRAM is much faster.

[00:17:08] You get about 1 millisecond,
so they are managed pretty faster.

[00:17:11] Famously, software used to be…

[00:17:15] Old-fashioned deterministic software,
the kind that's now out of favor,

[00:17:20] used to be very easy and quick to scale.

[00:17:23] You had a social networks that have some
Southwest, Northwest moment,

[00:17:26] and they can scale through 10, 100,
1,000 orders of magnitude of adding users

[00:17:32] because it's just a few rows
in a database, and it's

[00:17:35] a very underutilized CPU.

[00:17:37] What's interesting about the AI world

[00:17:39] is there are very real bottlenecks.

[00:17:45] Elon spent lots of time talking about
power, but it's not just

[00:17:48] bringing power online.
You mentioned HBM is reminding me of…

[00:17:51] It seems like there's a view that maybe
there's going to be some HBM

[00:17:56] supply chain crunch.

[00:17:59] Where do you see…

[00:18:01] Are we in for just a crunched world where
some limiter is

[00:18:07] pacing the rate of AI buildout over
the coming few years, where

[00:18:11] the economics work,
and the products and everything like that,

[00:18:14] but ultimately,
we just can't bring the components online

[00:18:17] fast enough because we have to build
out the factories, things like that.

[00:18:20] What are those crunched components?

[00:18:24] I think so.

[00:18:24] I'll just comment, by the way, this is a
great time to be a supplier in this place.

[00:18:29] Or just really— You should
have started an HBM company.

[00:18:31] I know.

[00:18:33] I think it's also just a fun time
to be someone who optimizes software.

[00:18:36] That's always what I like doing.

[00:18:38] Always the challenge is,
"Why am I optimizing this if no one

[00:18:41] cares?" But finally,
there's a place where you can…

[00:18:46] It's actually very meaningful
in a very tangible sense.

[00:18:49] If I can make this 20% more efficient,
then it can save that 20% of the buildout.

[00:18:54] The supply chain,
we're going to have crunches

[00:18:57] on all of the supply chain really.

[00:18:59] If you look at the big components of what
any company, like us for example,

[00:19:05] build out,
there is dependency on logic ties

[00:19:08] from typically TSMC, maybe Samsung, or
HBM, which are the big three HBM vendors.

[00:19:13] Hynix, Samsung, and Micron.

[00:19:17] Then there's also just the whole rack
manufacturing, which includes

[00:19:22] literally just sheet metal and so
on that builds the rack,

[00:19:24] but also cables and connectors because
of all the high-speed interconnect.

[00:19:29] That's what we—Racks don't sound hard.

[00:19:31] Are they sneaky hard?

[00:19:33] The big challenge is that you want
to bring in a huge amount of power,

[00:19:37] get a huge amount of heat out,
and also have phenomenal interconnect,

[00:19:41] which has very high signal
integrity requirements.

[00:19:44] Pack a lot of cables in with…

[00:19:46] The cables don't bend too much,
they have to have enough copper in them,

[00:19:49] and so on, that you don't lose
data rate on the interconnect.

[00:19:54] If you push it to a limit, it's sound.

[00:19:56] Wafers, racks, HBM.
What else?

[00:19:59] Data centers, which I think is power,
primarily a little bit of buildout,

[00:20:03] but primarily power
and good infrastructure there.

[00:20:08] How do you then,
as a startup that is looking to acquire

[00:20:12] all these components,
elbow your way in amongst the giants

[00:20:17] of the Googles and the NVIDIAs and all
these people who

[00:20:21] have long-running relationships
and have been buying for much longer?

[00:20:27] Ultimately what all of these suppliers
care about, they do somewhat care about

[00:20:31] a diversity of their own customers.

[00:20:34] It's not a great position to be in.

[00:20:35] They don't want monotony.
That's right, yeah.

[00:20:39] But then, what is their hesitation?

[00:20:43] Or the calculus for one of these large
suppliers is,

[00:20:47] if I reserve some of my capacity for you,
a startup, are you going

[00:20:51] to be around in a year?
Is anyone going to even buy your product?

[00:20:53] Our approach has been to
just actually find buyers for the product,

[00:20:59] and then the buyers answer
that question, ultimately.

[00:21:04] If you show up with a bunch of fairly
ironclad contracts to a supplier,

[00:21:08] then that has helped.

[00:21:09] That's the nature of it, yeah.

[00:21:11] I presume also
the round you just raised really helps

[00:21:15] there, where showing that you are
incredibly well-capitalized and not going

[00:21:20] anywhere also helps from a supplier
validation point of view.

[00:21:25] Absolutely.

[00:21:26] It helps just to say that we are around.

[00:21:28] We, in some cases, are actually…

[00:21:31] It depends on which part of the supply
chain, but some parts of the supply chain,

[00:21:35] some are fungible.
Logic ties are typically pretty fungible.

[00:21:38] But other parts of manufacturing,
you actually need something

[00:21:42] specifically set up for you.

[00:21:44] We're also able to cover
the capital costs of that.

[00:21:47] That makes sense.

[00:21:48] Coming back to the MatX architecture.

[00:21:51] You want to build the best chip for LLMs.
What is that?

[00:21:54] Yeah, exactly.
Sounds great.

[00:21:58] There's a few aspects to that.

[00:22:00] I think the first one is
pick your memory system right.

[00:22:05] I said, we've seen this HBM family,
we've got the SRAM family,

[00:22:08] put them both together is actually,
the most obvious idea,

[00:22:12] but you can actually do it.

[00:22:14] There are a lot of details
to make that work well.

[00:22:16] We've done that work.

[00:22:17] One of the things that shows up there is
you've spent all of this

[00:22:20] area on your chip on SRAM.

[00:22:22] How do you fit in the matrix multipliers,
which are the other big thing you really

[00:22:25] need to do,
and somehow create a much more

[00:22:28] efficient matrix multiply engine?

[00:22:31] There is a gold standard for that.

[00:22:32] That is called the systolic array.

[00:22:33] Make a really large systolic array.

[00:22:35] You can't beat that in area
or power efficiency.

[00:22:37] Provably so?
Practically?

[00:22:39] Practically.
It has not known a better approach there.

[00:22:43] The main thing is,
where are the inefficiencies typically?

[00:22:45] The inefficiencies show up when
you leave the systolic array.

[00:22:49] If you make a systolic array really big,
then you just don't leave it as often.

[00:22:54] That's the idea.

[00:22:56] Make a really big systolic array.

[00:22:57] That is sort of the theme of several
of the 2023-era startups, including us.

[00:23:06] But one of the challenges there is, now,
there is this part of the neural network,

[00:23:10] as part of the Transformer,
which is this attention that doesn't

[00:23:13] map well onto a large systolic array.
That's the tension.

[00:23:18] The mixture of expert layer maps really
well, but the attention does not.

[00:23:23] What we came up with,
which is quite different than some

[00:23:25] of the other startups in this space,
is say, take a really large historical

[00:23:29] array, but have a way to split it up
into pieces without losing efficiency.

[00:23:34] That is the core of the design for us.

[00:23:37] Then, the third component.

[00:23:39] First was HBM and SRAM,
second is the systolic array,

[00:23:42] third component is an interesting new
approach on low-precision arithmetic.

[00:23:48] Low-precision arithmetic, in general,
we've seen number formats

[00:23:51] get narrower and narrower.

[00:23:53] They get faster and faster as
you make them less precise.

[00:23:57] Number formats get narrower.
What does that mean?

[00:24:00] Float32 was how people
used to train neural nets.

[00:24:04] That's just too much precision.
It's wasteful.

[00:24:06] Too much precision, yeah.

[00:24:07] It's like saying,
"I've got an image with a billion

[00:24:12] color bit depth." It's too many colors.

[00:24:15] You'd rather have more
pixels and fewer colors.

[00:24:18] That trend seems to go
almost all the way down to one bit even,

[00:24:23] where just have very few colors
but a huge number of pixels.

[00:24:30] In net seems to be a better,
just more efficient way to train models.

[00:24:35] Sorry, literally what precision
are you dealing with in this sense?

[00:24:42] We have a range.

[00:24:48] We actually have an ML team who we hired
specifically to research

[00:24:52] different forms of numerics and how to
make them all work together really well.

[00:24:58] We have a range of precisions.

[00:24:59] It's not just one precision.

[00:25:01] We think probably the main thing will be
similar to where NVIDIA is at,

[00:25:04] which is 4-bit precision.

[00:25:06] But I think a mix of different precisions
is useful for just when you look

[00:25:10] at the research, sometimes you want some
layers in higher precision or

[00:25:12] lower precision, and so on.
4-bit is 16.

[00:25:15] Yeah, you get 16 choices.
That's it.

[00:25:17] That's it.
It's pretty imprecise.

[00:25:21] That's really interesting.

[00:25:21] I didn't know about that dynamic,
but it makes sense.

[00:25:24] Half of them are positive,
half of them are negative.

[00:25:26] It's even— How do you design a chip?

[00:25:33] Is that a whiteboard?

[00:25:34] What software are you working in?
I'd just love to know…

[00:25:37] I understand how you design software
and what that process looks like.

[00:25:40] I have actually no sense
for what chip design looks like.

[00:25:43] The way that you actually type a chip
into a computer is similar to software.

[00:25:47] You write Verilog.

[00:25:49] Verilog is a programming language.

[00:25:50] It is a very parallel programming
language, which makes it different

[00:25:53] than C or Python or something.

[00:25:56] But it is a programming language.

[00:25:59] The mechanics of how you express
the design are the same as software,

[00:26:02] and we have continuous integration,
Git, all of those things.

[00:26:05] But a program executes...

[00:26:07] Like your Verilog program...

[00:26:09] We don't really run it.

[00:26:11] Exactly.
How does it run?

[00:26:13] We synthesize it.

[00:26:15] Synopsys and Cadence provide EDA tools.
EDA?

[00:26:18] You have to remember, I'm just—
I don't even know what it means, really.

[00:26:24] I think it's Electronic Design Automation.

[00:26:27] It takes the Verilog and says…

[00:26:30] First turns it into a description of what
are the logic gates that are involved,

[00:26:33] ANDs, ORs, NOTs, and then
the wires between them.

[00:26:37] Then it runs for days doing some really
difficult algorithms,

[00:26:42] and then eventually produces…

[00:26:46] Gates are the first thing,
and then even below that,

[00:26:48] it literally just produces polygons.

[00:26:50] It says like,
P-type semiconductor here, N-type

[00:26:53] semiconductor here, and polysilicon.

[00:26:57] You write Verilog,
and then that compiles down into gates

[00:27:02] and ultimately the Minecraft 3D.

[00:27:07] "This is where your elements should go."

[00:27:11] But then, what is the iteration loop?

[00:27:16] When we write code at Stripe,
we build a first version of something,

[00:27:20] and then we try it out, and then we
refine it, and we add more

[00:27:25] functionality over time.

[00:27:26] We're going to write
some tests at some point.

[00:27:27] We'll ship that, we'll find product market
fit, and then we'll refine it in market.

[00:27:32] Do you just sit down and write the
completed chip, and it works really well?

[00:27:35] Every year we tape-out a chip,
and if there's a bug,

[00:27:37] we just wait till next year.

[00:27:38] It's not really how we do it.
What's the iterative process?

[00:27:41] How do we actually do it?
It's much more waterfall than software is.

[00:27:44] Waterfall is almost a bad
word in software development.

[00:27:47] But it's a fact of life in general.

[00:27:51] The waterfall goes from architects to
logic designers who are writing Verilog.

[00:27:57] Then, there's this design verification,
and then physical design.

[00:28:02] There's this really big architecture phase
which happens before even writing any

[00:28:05] Verilog,
which is, "What do I want the organization

[00:28:10] of my chip to be?" In some sense…

[00:28:13] I came to hardware after doing
almost 10 years in software.

[00:28:18] I really like the blank
slate you get in hardware.

[00:28:21] You've got all of the raw materials,
you have a much more

[00:28:24] varied in what you have available.

[00:28:27] What is the organization of your chip?

[00:28:28] Do I have 100 cores?
Do I have one core?

[00:28:31] Do I have systolic arrays?
Do I have vector units?

[00:28:33] All of those things.

[00:28:36] Then we spend a long time coming up
with that general principle

[00:28:39] and then saying, "Okay, now I've got
these applications I want to run.

[00:28:42] I want to run a transformer
of a particular shape.

[00:28:44] I want to map that onto this architecture
that I've got in my head."

[00:28:48] We do a lot of iteration.

[00:28:50] Well, I've got this
architecture in my head.

[00:28:51] I write it down to communicate to other
people, but that's just

[00:28:53] like a markdown file.

[00:28:57] Then, still actually a lot in my head,
but maybe with Python simulation and so

[00:29:01] on, I'll see,
do my applications map well to it?

[00:29:05] Can I run LLMs?

[00:29:07] You have a simulator where you
write your chip, you can then simulate its

[00:29:12] performance,
and you have some battery of tests

[00:29:15] that you see how this chip design works.

[00:29:18] Is it like an industry standard...

[00:29:22] Is it the X-Plane of chip testing?

[00:29:28] There's an industry standard thing for
the Verilog once you've done the design.

[00:29:32] They're just Verilog simulators
that you can test against.

[00:29:37] But you've already invested a huge amount
of work by the time you've got

[00:29:39] to that point, and so you sure hope you
haven't made a big mistake at that point.

[00:29:44] The thing that everyone does
prior to that is,

[00:29:49] we'll write our own performance simulator,
which, I mean,

[00:29:53] it is very specific to your particular
architecture, and you can write it quite

[00:29:56] concisely in just a normal
programming language.

[00:29:59] That is where most of the
architecture work is done.

[00:30:01] Then the simulation on Verilog is more,
"I know what I'm doing.

[00:30:03] I just want to make sure I didn't have any
bugs when I implemented it." But I presume

[00:30:07] it's a game of inches where different
people are trying different

[00:30:10] things, and then you...

[00:30:13] Do you simulate it to see if it runs
1% better across the battery of tests?

[00:30:18] Or is that not how it works?

[00:30:20] In this space, not so much.

[00:30:22] Just to characterize what performance
of an AI chip is, it is how many, really…

[00:30:29] First thing you care about is FLOPS.

[00:30:31] How many FLOPS have I got?

[00:30:32] That's a product of how many multiplies,
like, "I've got a grid of a certain size,

[00:30:37] 1000 by 1000." Can do a million
multiplies in a clock cycle.

[00:30:41] Then I have a certain clock
frequency, a gigahertz.

[00:30:43] I multiply them out.
That is the speed of it.

[00:30:47] I don't even need to write
that and test it to see how fast it is.

[00:30:50] It just is.

[00:30:52] What I plan in advance is
it's going to be this fast.

[00:30:55] What I can then optimize on,
maybe a little bit, is clock speed.

[00:30:58] There's not a lot I can do there.

[00:31:00] Then, I can optimize
a bit on area as well.

[00:31:03] There is some room for optimization,
but actually a lot of it gets set.

[00:31:06] Actually, just the speed of the chip
gets set very much upfront.

[00:31:11] Then how many chips do you fab?

[00:31:13] Is it only the ones going into production,
or is it just build a few to throw away,

[00:31:19] or how does it work?

[00:31:21] The ideal, which companies tend to hit
about 50% of the time,

[00:31:24] is that your first tape-out…

[00:31:27] Tape-out costs $30 million.

[00:31:28] Your first— Tape-out
is just production run?

[00:31:30] That's right.

[00:31:33] The actual manual, the first chip costs
$30 million, the second chip costs $1,000.

[00:31:37] Yes.

[00:31:38] Tape-out is that first chip.
Okay.

[00:31:40] The ideal is that your first tape-out
is actually your production thing.

[00:31:44] You do a tape-out, you
make maybe a thousand chips and test them,

[00:31:48] and then you do production volume.

[00:31:50] In the unlucky 50% of the time, you need
to redo some or all of your tape-out.

[00:31:58] In good cases, and in many cases,
you can redo just the metal layers

[00:32:02] which costs you only like $100,000.

[00:32:05] As opposed to the—
Pay the $30 million again.

[00:32:09] But in bad cases,
if you've made something serious and you

[00:32:12] can't fix it at the metal layers,
you have to do the whole thing again.

[00:32:15] Why can't that be solved?

[00:32:18] Is that definitionally an error
in simulation, where it turns out these

[00:32:21] two gates were too close together, and
it just led to some reliability issues?

[00:32:30] Yeah.

[00:32:30] What you're describing is physical,
the physical implementation

[00:32:35] of the chip is wrong.
That's one class.

[00:32:38] The other class is that the logical
specification of the chip is wrong.

[00:32:41] But shouldn't that be—
Shouldn't you have caught that before?

[00:32:44] Yeah.

[00:32:44] Before you spent $30
million on manufacturing it.

[00:32:46] Yeah.

[00:32:49] We do a lot of testing.
We try not to ship these things.

[00:32:53] I hear software companies also
ship bugs to production as well.

[00:32:56] Fair.

[00:32:57] Sometimes things— It's a very good retort.

[00:33:00] Shouldn't you not be shipping bugs?

[00:33:03] But there is a real trade-off in,
you can spend more and more

[00:33:06] time on design verification.

[00:33:10] There's always this question of,
when do you stop?

[00:33:14] You stop when your coverage metrics ever
hit a certain point, but maybe not 100%.

[00:33:19] Then if

[00:33:22] Apple has to discretize the iPhone release

[00:33:27] cycle, and they have settled on once per
year, they'll decide,

[00:33:31] "We've got this better camera,
but it's got to wait for the next

[00:33:34] version," or, "We're going to improve
the waterproofing, but that's got to wait

[00:33:38] for the iPhone 8 or whatever."
They have taken a continuous process

[00:33:42] of always coming up with ways to make
the iPhone better and discretized

[00:33:46] it into annual iPhone releases.

[00:33:48] What will your discrete cadence be?
What's our vision of that?

[00:33:52] Yeah.

[00:33:52] Many chip vendors have this sort
of tick-tock model, which is

[00:33:56] you'll do on one generation,
maybe you're trying to release every year.

[00:34:01] On even numbered years,
you'll do a physical technology upgrade,

[00:34:05] so new transistor technology, new
memory technology, and you interconnect.

[00:34:09] Then on odd numbered years,
you might do an architecture overhaul.

[00:34:12] I think that's a pretty good fit because
you have different parts of your company

[00:34:16] that are skilled at different areas,
and it allows you to keep both of them

[00:34:19] occupied without having instead every
2 years doing a massive risk release.

[00:34:25] Yeah.

[00:34:26] So you think that's
probably likely for you?

[00:34:28] That's right.

[00:34:29] You mentioned interconnect.

[00:34:31] There's a narrative out there
that in NVIDIA,

[00:34:34] a huge part of the defensibility comes not
from the chips, which are good,

[00:34:37] but from the software layer,
and the ability for engineers to write

[00:34:42] these really parallel workloads,
and the fact that they've been refining

[00:34:46] CUDA for whatever number of years.
A decade or so.

[00:34:49] Exactly, a long time.

[00:34:52] How do you think about parallelization,
and is that narrative true?

[00:34:56] It's true for sure.

[00:34:58] It's true in many areas of the market.

[00:35:03] I think,
and especially where you look at where

[00:35:06] NVIDIA entered the market, they're doing

[00:35:11] PC devices, lots of gaming, and so on.

[00:35:15] There are thousands of games,
maybe tens of thousands of games released,

[00:35:19] and they all need to be programmed against
CUDA, and so

[00:35:23] there's such a huge investment
in the software that this is really

[00:35:27] important, the compatibility.

[00:35:28] There are not thousands of LLMs.

[00:35:30] There's one LLM per frontier lab,
and there's maybe five frontier

[00:35:33] labs or something like that.

[00:35:37] Just the economics of that is different.

[00:35:39] The calculation for a frontier lab roughly
goes as, I just bought a $10 billion

[00:35:45] compute cluster,
I have hired 50 of the best

[00:35:50] people who can write optimized GPU,
or TPU, or Trainium software.

[00:35:56] I pay them less than
$10 billion, a lot less.

[00:36:01] Let's put them to work
optimizing the compute.

[00:36:07] Good work there,
depends on what your baseline is,

[00:36:10] but it can very easily double
the performance of the software you write.

[00:36:15] There is a huge amount of custom software
written for every generation of chip.

[00:36:19] When a new chip comes out, software is
substantially rewritten

[00:36:22] to optimize for that specific chip.

[00:36:24] That's just the right trade-off given
the relative costs of these things.

[00:36:28] What that means for us is
that ecosystem already exists,

[00:36:33] and that way of operating, where you say,
"I'm just going to staff a 50-person team

[00:36:39] to write software for this chip,"
works really well if you're

[00:36:42] trying to sell to frontier labs.

[00:36:44] You're saying CUDA is way more important
for the games environment,

[00:36:49] where it just does a lot of games
than this top-heavy AI market that we're

[00:36:55] in, where
if people say, "You need to then customize

[00:37:02] your workload for a MatX chip," it's like,
"Well, fine, do that." Cost of business.

[00:37:07] Yeah, that makes a lot of sense.

[00:37:11] Where will you fab the chips?
TSMC.

[00:37:16] Why is TSMC so durable?

[00:37:22] It's interesting.

[00:37:23] They don't charge a lot as well.

[00:37:24] You'd think that if they're a monopoly
provider, they should

[00:37:26] charge a lot of money.
They don't.

[00:37:29] I think that is a big aspect
of why they're so durable.

[00:37:32] It's like this
cyclical conservatism

[00:37:36] crossed with Taiwanese business
conservatism, means you're at the most

[00:37:41] conservative part of the matrix.

[00:37:43] But I mean, it does…

[00:37:48] I mean, an American capitalist might say,
"Well, they're just screwing up.

[00:37:51] They could have extracted more money
from the market." But you could also say

[00:37:54] that this is actually the long-term
sustaining advantage because they will

[00:37:59] just stay ahead for a really long time.

[00:38:01] They don't incur
the creation of competitors.

[00:38:03] But isn't the creation of competitors
priced in because

[00:38:06] of the geopolitical risk?

[00:38:10] It's not like everyone's fat, dumb,
and happy with their TSMC dependence.

[00:38:13] They're actually thinking a lot about it.

[00:38:15] So there is real technical
advantage there as well.

[00:38:17] It's not just the discouragement.

[00:38:19] But designing chips seems really hard,
building airplanes seems really hard.

[00:38:23] There are so many areas where competitive
market forces create multiple options.

[00:38:31] Yet, that has not occurred here.

[00:38:34] There are multiple options.

[00:38:35] You can buy from Intel or Samsung.

[00:38:37] But at leading-edge nodes.

[00:38:40] What do we even care about
in leading-edge nodes, I guess?

[00:38:42] The big advantage is on power.

[00:38:44] The advantage on area is smaller.

[00:38:46] The leading edge nodes, the density
doesn't go up as much as it used to.

[00:38:52] When you are really,
really sensitive to power, it is a good

[00:38:54] idea to be on leading edge nodes.

[00:38:56] That is AI chips and mobile phone chips.

[00:38:59] But there's a lot of the market where you
don't actually like

[00:39:02] the devices in your car.
Sure.

[00:39:04] Car chips, yeah, that's fine.

[00:39:05] But you're saying,
if you exclude the two most interesting

[00:39:09] parts of the market— That's true.

[00:39:16] For this super high-growth area
of the market, it's interesting to me,

[00:39:17] again, there's a lot of other
really complex business problems out

[00:39:22] there that competition has solved.
Chip design is a…

[00:39:25] Like, why has someone not left
TSMC and gone and built a new fab?

[00:39:29] I don't know.

[00:39:32] The cost of a fab is extremely expensive.

[00:39:35] I recognize that also the cost
of a lab is extremely expensive, too.

[00:39:41] I don't really understand the technical
details of why it's so hard.

[00:39:45] I mean, there is some amount of just a $10
billion fab versus $100 million

[00:39:51] tape-out and chip development.

[00:39:52] There's a huge difference there.

[00:39:53] But beyond that, I'm not sure.

[00:39:55] What's TSMC like to deal with?
They're very big.

[00:39:58] As a startup, we tend to work with,
not directly with TSMC, but with

[00:40:04] an ASIC vendor who,
firstly, does a huge amount of the actual

[00:40:07] backend work for us to interface
with them, but then also has their

[00:40:10] existing relationships with them.

[00:40:13] TSMC cares a lot about diversity
of their customer pool.

[00:40:18] It gets back to that conservatism.
Yeah.

[00:40:21] They're great to work
with from that perspective.

[00:40:24] They want to encourage startups.
That's right.

[00:40:26] That's very cool.
Why don't the labs design their own chips?

[00:40:29] Google does.
Google does.

[00:40:31] OpenAI is starting.

[00:40:32] It's really a trade-off of how much
advantage do you get from vertical

[00:40:35] integration versus how much advantage
do you get by concentration of R&D work.

[00:40:41] You take the five labs,
and if they all buy from one player,

[00:40:43] then you can put five times
as much R&D into that chip.

[00:40:48] Does that beat the advantage you get from
saying, "I know exactly what my model is"?

[00:40:52] Because of the several-years delay
from designing a chip to being

[00:40:57] in production,
you can't actually say,

[00:40:59] "I know exactly what my model is," because
models change much faster than that.

[00:41:06] Even the labs are forced into this
position where they have to make

[00:41:09] predictions, and they have to hedge
against what they might

[00:41:11] do two years from now.

[00:41:12] The calculus is, what is the probability
distribution of what my model might look

[00:41:16] like and then
design a chip that gets 90% of that

[00:41:20] probability distribution or something?

[00:41:23] Elon is excited about
data centers in space.

[00:41:27] The two criticisms I've heard are that
cooling is very hard and then

[00:41:34] repairing the chips is hard.

[00:41:37] But I know nothing about chips.
You do.

[00:41:41] The repair is really interesting.

[00:41:43] When you look at how NVIDIA deploys their
racks, how we do something pretty

[00:41:47] similar to what NVIDIA does.

[00:41:50] In general, you always need to design
for the fact that some of your

[00:41:52] chips are going to be down.

[00:41:53] Mean time between failure
of chips is not that large.

[00:41:58] In a cluster of 100,000 chips,
there are going to be chips

[00:42:01] that are down all the time.

[00:42:02] One way you can do that is you can make a
rack where one rack has spare chips in it.

[00:42:07] NVIDIA has eight spare
chips in a rack of 64.

[00:42:13] That's pretty good.

[00:42:15] The combinatorics works
really well for you there.

[00:42:20] Because you can pick which ones to avoid,
you can with very high probability

[00:42:27] tolerate a lot of failures.

[00:42:27] And then the other just for the other
family of things is to say my rack has to

[00:42:31] work, but I have some spare racks as well.

[00:42:35] You can math that out with
the tax of reliability here is

[00:42:38] only like 10%, that's pretty good.

[00:42:41] But that relies on someone coming
and servicing the part within

[00:42:45] a day or something like that.

[00:42:46] If you say they're going to service it
never, then

[00:42:50] I think you actually can get where you
want to be, but maybe with 100%

[00:42:53] tax on reliability rather than 10%.

[00:42:55] For example, if you think the average
lifetime of a chip is in the range

[00:42:59] of three to five years.

[00:43:01] That means if I deploy twice as many
chips, then three to five years from now,

[00:43:05] half of them will still work.

[00:43:06] Also, the burn-in is
particularly failure-y.

[00:43:10] How about the cooling?
Most of the challenge...

[00:43:14] I guess this is actually really
a data center design aspect.

[00:43:18] At the rack level,
the challenge of cooling is just getting

[00:43:20] the heat out as quickly as possible out
of the rack into the cooling network.

[00:43:26] How you get it out of the spaceship,
other people would know

[00:43:30] that better than I do.

[00:43:33] That seems to be the main
objection, but I don't know.

[00:43:37] I think it's
if you think the cost of repair is

[00:43:40] that you need to have deployed twice as
many chips, then it's a trade-off

[00:43:43] of the capital of the chips
versus the power saving.

[00:43:46] Exactly, the repair thing, it feels like,
can be solved, because also,

[00:43:51] I think part of the bet,
part of Elon's claim,

[00:43:54] is that we will just be so power-limited
that you have no option

[00:43:58] but to go to space.

[00:44:00] People can argue about that,
but were that to be the case, then yes,

[00:44:05] it's like you can get power in space,
and you cannot on Earth,

[00:44:08] so you might as well go there.

[00:44:10] Whereas like the cooling is a more
fundamental, "Does the product actually

[00:44:13] work at all?" Reiner thinks about AI

[00:44:19] the unglamorous way: compute,

[00:44:20] systems architecture, and what it
takes to run models reliably at scale.

[00:44:24] If you're building an AI product,
the business model similarly has

[00:44:27] a ton of unglamorous complexity.

[00:44:29] You're not just selling AI,
you're monetizing consumption across

[00:44:32] API calls, tokens processed, GPU hours.

[00:44:36] Stripe Billing is a scalable
system for usage-based billing.

[00:44:39] It lets you launch token-based pricing,
subscriptions, credits,

[00:44:41] hybrid models, whatever you want.

[00:44:43] You can create revenue models based
on usage without rebuilding your

[00:44:46] pricing system every six months.

[00:44:48] If you're building an AI product,
Stripe Billing is worth a look.

[00:44:55] What are your AI predictions for 2026?

[00:44:58] What I'm really excited about is
just being able to…

[00:45:04] I'm still excited about the coding.

[00:45:06] This is what we do as a company.

[00:45:07] It's what many others
do as a company as well.

[00:45:10] The one aspect of this is
expanding into more domains.

[00:45:16] For example, in where we spend our time.

[00:45:21] We as a company, we write Rust,
we write Verilog, we write Python.

[00:45:26] No Haskell?
No, there's a story there.

[00:45:28] I used to love Haskell.

[00:45:30] Rust is my current favorite.

[00:45:32] Mutation is good.

[00:45:36] The models are extremely
good at Rust and Python.

[00:45:40] They've done a lot of RL on them.

[00:45:42] They have not done as much RL on Verilog.

[00:45:46] They've done almost none on,
"Write me a markdown file that describes

[00:45:51] a chip architecture."
How do you even RL on that?

[00:45:55] You have to say,
what is a good chip architecture?

[00:45:58] You have to somehow say whether
that's a good result or not.

[00:46:02] I think one of the things the labs are
doing is trying to broaden what they've

[00:46:05] done RL on, source it from customers

[00:46:08] and so on in order to make it less spiky,

[00:46:15] fill out the gaps between the spikes.

[00:46:17] I presume the labs would love to work
with you on improving the models

[00:46:23] by doing RL on this specific task.

[00:46:28] However, it's also special.

[00:46:29] How does that make sense for us?

[00:46:31] You're a special sauce.

[00:46:33] Do you want to come up with some AI
approaches but keep them proprietary?

[00:46:40] We've looked at a few
different aspects here.

[00:46:42] There's the…

[00:46:43] What we're able to do by ourselves,
our business is not training models.

[00:46:46] We do it in order to do the research
on numerics, but actual

[00:46:50] production models we don't do.

[00:46:54] The biggest mileage is on the RL, and it's
not something we can really do ourselves.

[00:46:59] We'd love it if we could have a custom
model just for us, but that doesn't

[00:47:02] seem to be— Well you could, right?

[00:47:05] The terms we've been offered by labs so
far have not been on those terms, but—

[00:47:08] Because you have to share the IP back.

[00:47:10] The way they prefer to do it is that
they put it into their mainstream

[00:47:15] model because it's good for them.

[00:47:17] Which, obviously, you don't want to do.

[00:47:19] How do you think…

[00:47:21] What does you using AI to design
a model do you think look like?

[00:47:25] Because this is actually an interesting
sight glass into a weak version

[00:47:29] of recursive self-improvement where we're
using the AIs to develop better AIs.

[00:47:35] I'm curious,
what you think that looks like?

[00:47:37] Is it your own proprietary
recursive models?

[00:47:40] What else?

[00:47:41] Is there day-to-day AI
usage that's load-bearing?

[00:47:45] The stuff that is available today and I
think will become even better,

[00:47:49] very quickly is just
the stuff that looks most like software.

[00:47:54] Writing Verilog, running tests,
running continuous integration, and so on.

[00:48:00] That is a big fraction
of the development time in a chip.

[00:48:03] It's probably 9, 12, 15 months.

[00:48:05] There's some stuff that's downstream
of that, which is physical design,

[00:48:10] which is, you take that Verilog and
you generate the gates and the polygons.

[00:48:17] We don't have a clear path for…

[00:48:19] It's not at least the most obvious thing
is not clear for how to compress that.

[00:48:24] The goal, can you tape-out
a chip in one month?

[00:48:26] One month would be the goal.

[00:48:28] In theory, you could compress all
of the logic design and design

[00:48:30] verification down to a short amount
of time just by continuing

[00:48:35] on the same path we're doing now.

[00:48:36] But if you wanted to take
the physical design down,

[00:48:39] that has to leave code,
you're now doing like graphical interfaces

[00:48:42] and saying I want to play stuff and so on.

[00:48:45] Actually, there has been work on this even

[00:48:47] prior to LLMs, which is a specific

[00:48:53] model trained for that particular problem.

[00:48:56] I think the vendors,
which is like Synopsys and Cadence,

[00:49:00] probably should move in that direction.

[00:49:03] Most of the focus has not been,
do it faster; it's been,

[00:49:05] do it with higher quality.

[00:49:08] But that is a big bottleneck on,
can I have a new chip every month?

[00:49:13] There's just the practical thing of a new
chip every month doesn't really make

[00:49:16] sense because then if I'm deploying...

[00:49:19] If it takes me a year to populate a data
center, that means I'm going to have

[00:49:21] different chips in different
corners of the data center.

[00:49:23] Yes.

[00:49:24] When you talk about one month to tapeout,
so you do all this work

[00:49:28] to ultimately produce a file.

[00:49:34] Everything TSMC then does,
it's not entirely in software.

[00:49:38] Is there some type-setting that has
to happen of moving stuff around?

[00:49:42] But what happens when you
send your files to TSMC?

[00:49:45] Then what?
They create a mask.

[00:49:47] That is where the ASML tools come in.

[00:49:52] A mask is really just a stencil.

[00:49:55] You shoot the lasers through the mask or
the x-rays through the mask and then

[00:50:00] that produces the different
P-type and N-type semiconductors.

[00:50:05] They produced the mask.
That is the expensive part.

[00:50:11] They're building up these
15 or so metal layers.

[00:50:14] They placed on the silicon and then
there are different layers of metals

[00:50:17] which connect all
the transistors together.

[00:50:20] They do that on a wafer.

[00:50:22] It happens on a stepping basis.

[00:50:25] There's a maximum size of chip you can
build which is constrained

[00:50:28] by this machinery.

[00:50:30] The wafer stepper is part
of the ASML special sauce, right?

[00:50:34] There are probably some important
alignment requirement there.

[00:50:38] I think I remember that being quite like
the classic manufacturing

[00:50:41] throughput problem.

[00:50:42] I think they've done a lot
of work on optimizing that.

[00:50:45] They tape that, so then you just
produce hundreds of copies of your chip.

[00:50:49] You have to test it
because there are defects.

[00:50:52] You typically, I think the average rates
really depends on process and so on,

[00:50:58] but small single digit
number of defects per chip.

[00:51:03] You test the chip and see
whether it has any defects in it.

[00:51:06] Many chips are designed to be able
to tolerate a few defects, and so you need

[00:51:10] to configure it to tolerate the defects.

[00:51:12] Now you have a die that by itself works.

[00:51:15] Then you need to package it.

[00:51:16] You put it
in a package together with memories,

[00:51:19] typically that's the HBM, and then maybe
you escape the wires

[00:51:24] to connect to other chips.

[00:51:25] How long does it take to make a mask?

[00:51:29] What we see is time from tape-out
to first chips back.

[00:51:34] Again, depends on node,
but it's ballpark four or five months.

[00:51:37] So tape-out is just sending the file?

[00:51:42] We consider tape-out as sending the file,
and then there's a whole process of you

[00:51:45] make the masks for all the layers, and
then, actually just producing the chips.

[00:51:49] Producing the masks and producing
the chips happens after tape-out.

[00:51:51] That's right.
I see.

[00:51:53] Is the term tape-out from like you send
a magnetic tape with the

[00:51:56] instructions or something?
It could be.

[00:51:58] I was in software when the
software was created.

[00:52:02] I was curious about
the tape actually means.

[00:52:04] It feels like…

[00:52:04] When I think about AI predictions,
one thing I'm really struck by is how,

[00:52:11] still in 2026, every time you open a

[00:52:16] chat window, it's contextless.

[00:52:19] It's got no memory.

[00:52:20] Now, to be fair, it's like,
guys, it's been four years.

[00:52:23] Not even four years.
It's been three and a half years.

[00:52:25] Just calm down, we'll get there.

[00:52:27] But I also interpret
a lot of the current enthusiasm

[00:52:32] for OpenClaw and all that stuff as

[00:52:36] this super hacky backdoor into state

[00:52:40] management where your little claw will
write a markdown file of what it's doing

[00:52:45] and then look at that markdown file
the next time and things like that.

[00:52:48] But it just feels like
state management and memory is going to be

[00:52:53] a huge deal and that will really
change the character of AI products.

[00:52:58] It's really interesting.

[00:53:01] Long context is one of the biggest
bottlenecks on speed of the model.

[00:53:09] Every single token you generate,
it reads through all of the previous

[00:53:12] tokens, or maybe it reads through a subset
of them, but reads through a lot

[00:53:15] of the previous tokens you've written.

[00:53:19] Memory bandwidth for that is
really constraining.

[00:53:21] You can think of model-level ways to solve
that problem, which is to say

[00:53:26] maybe I can compress it into fewer
bytes or something like that.

[00:53:29] But it's interesting that the most
effective way to solve it has been—it's

[00:53:33] really a combination of everything—but
the most effective way to solve it has

[00:53:36] been
once you hit your 300,000 token limit,

[00:53:40] have the model go back
through it and compact.

[00:53:44] This is kind of what OpenClaw is doing.

[00:53:46] It's compacting everything you've done.

[00:53:49] But it's funny that it's so manual.

[00:53:53] I think— Manual is the wrong word.

[00:53:55] It's so primitive.

[00:53:57] It's maybe because it's so controllable.

[00:54:00] If you want to iterate on how you compact,
you give a different prompt, and you say,

[00:54:04] "Compact this way, compact that way." You
can iterate that on that

[00:54:06] in seconds or minutes.

[00:54:08] Whereas if you're trying to do some
iteration on the model level,

[00:54:11] where you say, "Now I've got a different
model architecture," it's going to take

[00:54:14] you months to train and launch something.
Yes.

[00:54:17] Any other AI predictions?

[00:54:18] I'm generally just interested in what
makes models cheaper and faster.

[00:54:25] Just at the model architecture level,
really tied into this context thing.

[00:54:28] I think the context size will stay
ballpark the same where it is,

[00:54:32] maybe a few times larger,
but the parameter count will go up.

[00:54:36] Parameter count should grow much faster
than context length actually,

[00:54:38] just because of the underlying
physics of what's available.

[00:54:42] Though, has that been the story?

[00:54:46] Would that be a reacceleration
of parameter count?

[00:54:48] Because it feels like we've leveled off
slightly in the last year or two,

[00:54:52] and instead we've been
focusing on more and better RL.

[00:54:56] Parameter count or
thinking tokens, I guess.

[00:55:00] Those are available, but the context
length is struggling to grow.

[00:55:07] But you think we…

[00:55:08] When you say context length is struggling
to grow, but you're saying we

[00:55:10] keep context the same length.

[00:55:12] Keep context the same length.

[00:55:13] But we're better at working
with large context.

[00:55:15] Is that what you're saying?

[00:55:17] Just have application-level interventions
to manage large context, like compacting.

[00:55:22] Because I think everyone's had
the experience

[00:55:24] currently of the chat conversation
and the further down in the chat you get.

[00:55:28] It just gets looser and—Yeah, sloppy.

[00:55:30] It's just like really sloppy by the end,
and it's like making mistakes, whatever.

[00:55:33] You're saying we start to do
better with large contacts.

[00:55:36] I buy that.

[00:55:37] When will I be typing into a chat window,
and it is a MatX chip

[00:55:43] underneath it, powering it?

[00:55:45] Tape-out in under a year.

[00:55:47] That means chips available end of year.
Ballpark.

[00:55:52] That's exciting.

[00:55:52] So in 2027, I will be seeing very high
performing chats as

[00:55:56] a result of your chips.

[00:55:57] In the 1% experiment of the users
or something like that.

[00:56:00] Exactly.

[00:56:00] I need to find a way to finagle
myself into the A/B test.

[00:56:04] MatX is 100 people?
That's right.

[00:56:09] How have you gone about
building the team, the culture?

[00:56:16] What we have on the team is hardware,
mostly hardware, but a big software

[00:56:20] team and also a big ML team.

[00:56:23] I think the ML team is quite
unusual in what we ask them to do.

[00:56:27] When you look at a typical ML team
in an AI chip company, it will be

[00:56:33] what I might say ML
engineering or ML performance.

[00:56:37] They're writing kernels that actually
just use your hardware

[00:56:43] as well on a given model.

[00:56:44] There's a missed opportunity there.

[00:56:45] If you're saying all we do is we take
other people's models,

[00:56:48] and we write kernels for them,
you're optimizing this, but you

[00:56:52] can't optimize this at the same time.

[00:56:53] We want to optimize the whole thing
at the same time, real code design.

[00:56:57] Our ML team is actual real ML research.

[00:57:00] What they do every day is they train
small LLMs from scratch focusing

[00:57:07] on numerics and attention.

[00:57:12] This has really, really helped us
make an interesting product.

[00:57:18] It's shown up most
strongly in our numerics.

[00:57:23] Often what you see when people design
numerics is they say,

[00:57:27] back when Float32 was popular,
it would be, "I'm going to follow the IEEE

[00:57:30] standard." Now it is like,
follow the Open Compute standard.

[00:57:34] There are lots of little details where you
say things like

[00:57:38] maybe, "What's the rounding mode I'm going
to use?" Like,

[00:57:41] "Round to the nearest even," or something
like that, which is like

[00:57:45] the best known standard for how to round.

[00:57:46] We want to cut corners anywhere we can.

[00:57:49] Maybe don't do the best rounding, maybe
don't do the…

[00:57:54] Don't get all the corner cases correctly.

[00:57:56] That's a very scary proposition if you're
just making those choices blind,

[00:57:59] but if you have the benefit of a research
team who can back you up as you do that,

[00:58:05] it's really powerful,
and it's really interesting that we can

[00:58:08] make some sloppy choices in these cases.

[00:58:11] I feel like often technical advances
come through better iteration loops.

[00:58:17] A favorite example of this I found
recently was that the Wright brothers

[00:58:20] actually had a failed
season before first flight.

[00:58:25] First flight was in 1904,
and they were down in Kitty Hawk in 1903

[00:58:29] and not making that much progress.

[00:58:31] They went back
to Ohio, and they had a wind tunnel,

[00:58:34] and they were testing their
design in the wind tunnel.

[00:58:37] You can imagine not a lot
of wind tunnels in 1904.

[00:58:42] They did a lot of wind tunnel testing and
their successful flight was after that.

[00:58:46] Is this something you're focused on where,
to get better chips,

[00:58:50] you allow for a better
testing and iteration loop,

[00:58:53] and what does that look like?

[00:58:56] I think this mostly happens in the
architecture and product definition stage.

[00:59:02] Maybe even more generally, I think
AI chips seem to live or die

[00:59:06] by product definition and architecture.

[00:59:08] What is the most extreme form of fast
iteration is doing it in your head.

[00:59:12] Can you map a model
to hardware in your head?

[00:59:15] Can you estimate the performance
of what it is in your head?

[00:59:18] You're not going to be 100% perfect,
but maybe you can prove some

[00:59:21] lower bound in performance.

[00:59:24] The simplest possible thing is my model
has a trillion parameters, my device

[00:59:31] can do a billion multipliers per second,
so it takes 1,000 seconds to run or

[00:59:34] something like that,
just to do that simple division.

[00:59:37] But then there are much more complicated
things like we tend to look at

[00:59:40] resource balances,
like how many memory fetches do I need

[00:59:43] to do per a multiplier
or something like that.

[00:59:49] At least the way I like to do design
and architecture and optimization is to

[00:59:55] be able to estimate the performance
to within about 30-40%

[00:59:59] before even typing anything in at all.

[01:00:02] We've tried to do that a lot.

[01:00:05] A lot of our architecture
comes from there.

[01:00:08] Then the next stage of iteration is…

[01:00:11] That's on the performance side.

[01:00:13] This also happens
on the circuit design side as well.

[01:00:16] Can you take a circuit and say
what is the gate count on that?

[01:00:20] A 16-bit multiplier has approximately 16
squared mini-gates,

[01:00:24] and you can do that for more complicated
things by sorting networks and so on.

[01:00:29] We already have a pretty good idea
of the costs and speeds of things at that

[01:00:33] point after doing these calculations.

[01:00:36] Then what we tend to do as
the next step of iteration is

[01:00:40] on the ML side, we run model experiments.

[01:00:42] You get iteration speed just
by having small models mostly.

[01:00:48] On the hardware side,
we use performance simulators to do

[01:00:53] the next level of detail to make sure
we're seeing all

[01:00:55] the things we want to see.

[01:00:56] This idea that the best iteration is
in your head is reminding

[01:01:01] me of Jeff Dean's numbers...

[01:01:04] Do you have your equivalent
of that numbers every MatX?

[01:01:06] We have go/gates in our company,
which says, what is the cost of an XOR

[01:01:11] gate, an AND gate, a full adder,
SRAM bitcell, and so on.

[01:01:15] You want people to be working
with that stuff in their head and have

[01:01:17] an intuitive sense for it because
it leads to better iteration.

[01:01:22] What is the pitch to someone joining MatX?

[01:01:26] I think if you are someone who likes
optimizing,

[01:01:30] just optimize something—software,
hardware, factorial,

[01:01:36] whatever—if you're trying to fit something
into the smallest budget possible, I

[01:01:41] think it's a pretty exciting place to be.

[01:01:44] I think hardware companies in general are
really exciting because you have such a

[01:01:50] broad range of skills
of people on the team.

[01:01:52] You have software people,
you have hardware people,

[01:01:54] you've got physical design,
you've got people who are just looking at

[01:01:57] the insertion force of a card into a rack.

[01:02:02] So there's so much discussion
and learning you can do.

[01:02:06] I think MatX in particular,
we really care about this and I think we

[01:02:11] extended it all the way up
into the application

[01:02:13] and the machine learning as well.

[01:02:20] Really, really interesting technical
problems, and I think just generally

[01:02:25] there's lots of interesting
people to talk to.

[01:02:26] Yes.

[01:02:27] Presumably in terms of impact,
if you can design a meaningfully higher

[01:02:32] throughput chip, a 20% higher throughput
chip means 20% more AI is happening.

[01:02:37] If the bottleneck is elsewhere, like
power or something like that or cost,

[01:02:41] you actually just are
meaningfully increasing the amount

[01:02:44] of intelligence in the world,
which is presumably exciting to people.

[01:02:49] I think this shows up both as just
can it apply in more applications,

[01:02:54] as well as just how smart is the model.
Yes.

[01:02:57] How about Rust?

[01:02:59] A previous project I worked on at Google,
we did a lot of Haskell.

[01:03:04] I did Haskell when I was at school.
I loved it.

[01:03:08] Very principled, very interesting.

[01:03:10] I like Haskell, but I also
like making stuff fast.

[01:03:14] The question is,
what is the first thing you want to do?

[01:03:17] You want to be able to modify your memory.

[01:03:19] Haskell, you jump
through hoops to do that.

[01:03:21] Maybe I just want a language that is
functional programming

[01:03:24] that lets me modify my memory.

[01:03:26] I think Rust has a lot of the nice things
which are like type classes or traits

[01:03:32] and a rich type system.

[01:03:35] One of the things that we have done,
interesting ways we use it at MatX are

[01:03:41] the range of data types
that you express on software.

[01:03:45] What are the integer types?

[01:03:46] Int32, int64, int8,
maybe that's all you care about.

[01:03:50] But it turns out in hardware,
you care about every single bit, and so

[01:03:55] you want to use 17, 18, 19-bit integers.

[01:04:01] That is quite natural to express,
and we build up a whole ecosystem of

[01:04:07] rich hardware data types in Rust as well.

[01:04:11] Has Rust beaten Go for the position
of performant type programming language

[01:04:17] with modern features or do they
actually address different?

[01:04:22] There's what the Rust marketing will say,
which is "safe without a garbage

[01:04:28] collection," which I think is a real…

[01:04:31] It is the objective thing that you can say
is different, but sort of buries the lead,

[01:04:37] which is also just it's got nice type
system features that Go doesn't have.

[01:04:42] Why is garbage collection…

[01:04:44] Why does it matter at all?

[01:04:46] People often focus on the time it takes
to run a garbage collector,

[01:04:49] but the other thing is that every time you
allocate an object, you've got the object,

[01:04:52] and then you've got the garbage
collector header at the beginning.

[01:04:55] It uses a lot more memory as well.

[01:04:57] If you want to design some,
I don't know, data structure that uses

[01:05:01] the right amount of memory
rather than a bit more than…

[01:05:04] I hadn't realized that in Rust you're
allocating your memory manually

[01:05:07] versus in Go you have— That's right.

[01:05:10] I didn't realize that.
And you prefer that for what you're doing.

[01:05:13] I just really like
dealing with the details.

[01:05:15] Like, you give me a puzzle,
and I'll be like,

[01:05:16] "Let me solve every single piece of it."
That tickles that part

[01:05:20] of my mind with Rust.

[01:05:21] It seems like you're a fan
of optimization generally.

[01:05:24] Is that a fair characterization?
Yeah.

[01:05:26] Where else have you…

[01:05:27] Chip optimization is one
domain, where else?

[01:05:36] When one of the really exciting things I
found about working at Google is

[01:05:39] that the whole Google code base is
available, and you can look at

[01:05:43] how does a memory allocator work,
how does a mutex work,

[01:05:46] how does a HashMap work,
any of those things, and you can go

[01:05:49] and look inside the implementations.

[01:05:51] Google has excellent implementations of
those, some of the best you could write.

[01:05:59] One of the things I did on my nights
and weekends when I was at Google was just

[01:06:02] to go find those implementations,
write a benchmark.

[01:06:07] How many nanoseconds does it take
to allocate eight bytes of memory?

[01:06:11] Can I make that faster?

[01:06:13] Can I maybe I inline this function?

[01:06:15] Maybe I look at the assembly and say,
"It looks like

[01:06:18] there's a few memory moves here or
there are some registers that are being

[01:06:21] used that I don't need in the
fast path and in the slow path.

[01:06:25] Can I do something there?"
I don't know, that was always my

[01:06:30] just fun and learning activity.

[01:06:32] Being outside of Google, I feel,
I probably could have done this inside

[01:06:36] of Google as well, but outside of Google,
I felt the luxury to be able to talk about

[01:06:42] these results as well.

[01:06:43] One of the things I've looked at recently
is just hash tables are used so much.

[01:06:51] One prompt for me was,
if I wanted to design custom CPU

[01:06:57] instructions for accelerating hash tables…

[01:06:58] Hash tables are one
of the most common things.

[01:07:00] I'm looking at them up
and writing them all the time.

[01:07:02] What would the optimal CPU be for that?

[01:07:07] Following down that chain is like,
what is the best hash table

[01:07:12] implementation in the first place?

[01:07:14] I spent some time looking at
different SIMD implementations.

[01:07:18] There's this really cool technique called
'cuckoo hashing' where you

[01:07:22] hash into two different locations,
and then you use the bucket

[01:07:26] which is less full.

[01:07:30] It's been in the literature for decades
and yet the best hash table

[01:07:35] implementations don't use it
because it's somehow not practical.

[01:07:39] Why is it not practical?

[01:07:40] Practical hash tables are
these days considered to be ones that use

[01:07:46] SIMD vector instructions to scan
eight buckets at a time.

[01:07:54] The way cuckoo hashing is normally
described is I look up one

[01:07:57] bucket here and one bucket there.

[01:07:59] I'm not using the vector instructions.

[01:08:00] Vector instructions are much faster than
scalar instructions and so

[01:08:03] there's a missed opportunity.

[01:08:05] Again, just
take the two good ideas and stick them

[01:08:08] together, do vector
instructions on cuckoo hashing.

[01:08:13] You have to be careful to get the details
right, but if you get it right,

[01:08:15] you can actually just wing it.

[01:08:16] Sorry, is your claim that one could design
a custom CPU that has way better hash

[01:08:22] table performance,
or even on current chips, you could

[01:08:26] get way better hash table performance?

[01:08:28] Both.

[01:08:31] I'm interested in what you can design
in custom hardware,

[01:08:33] but MatX doesn't make CPUs.

[01:08:35] We're not going to make CPUs.
You could.

[01:08:36] New line of business.

[01:08:39] We just want to focus on shipping
one product well for the time being.

[01:08:43] Fair.
Good answer.

[01:08:45] I think it's an interesting exercise,
but I don't get to feel the endorphins

[01:08:49] of seeing the number go down.

[01:08:52] I first did this on just Intel CPUs.

[01:08:55] You can get better performance
than some of the best hash table

[01:08:59] implementations available
using cuckoo hashing on Intel CPUs.

[01:09:05] What are examples of workloads that are
really hash table read intensive?

[01:09:11] I know everything,
JavaScript, I guess, but…

[01:09:18] It's a tricky exercise because when you
really think about it, you're like,

[01:09:22] "Did I really need a hash table there?

[01:09:23] I probably didn't, but I just reach for it
all the time." But you can go

[01:09:26] to the Google JavaScript team and probably
help them eke out better performance

[01:09:30] in the Chrome JavaScript engine?
Potentially.

[01:09:33] I'm not going to spend my time on that.

[01:09:35] If they're listening to this podcast,
just a free idea from Reiner.

[01:09:39] Explain the dragon.

[01:09:41] This is from a book that,
when I was working on the JAX team…

[01:09:44] The JAX team is one of the ML
infrastructure teams at Google.

[01:09:47] I was there as the most
recent team before I left.

[01:09:50] I'm sorry, what does the JAX team do?

[01:09:52] The JAX team develops…

[01:09:54] This is Google's new, more modern version
of TensorFlow or a competitor to PyTorch.

[01:10:00] It's how you write models
in Python to run on TPUs.

[01:10:03] A big part of the JAX team, however, is to
say, we have JAX, the technical artifact.

[01:10:08] Can we help enable users to actually use
it really well and get high performance?

[01:10:14] Ultimately that became, who are the users?

[01:10:17] It's people writing LLMs.

[01:10:18] How do you get good performance on LLMs?

[01:10:20] Really, really strong team,
the JAX team at Google,

[01:10:24] although as with a lot of brain
people are now elsewhere as well.

[01:10:28] We developed a lot of the different
techniques for how to lay out

[01:10:31] models efficiently on many chips.

[01:10:34] Ultimately, some people at Google—and I
contributed after I left Google—wrote this

[01:10:39] guide called How To Scale Your Model,
how to run an LLM as fast as possible.

[01:10:44] It is the main reference for how
to get high performance on TPUs.

[01:10:47] There is now also a GPU
version of this as well.

[01:10:49] It's a dragon because it's
How to Train Your Dragon.

[01:10:52] I see.

[01:10:55] Last question.

[01:10:57] People might not have thought that
there's room for new chip companies.

[01:11:02] It might have seemed unusual or very hard.

[01:11:05] You guys, it seems like a very
good approach with that.

[01:11:09] Where do you think are other opportunities
for companies to be started here in 2026?

[01:11:14] Where do you think people should be
looking for entrepreneurial opportunities

[01:11:17] or just technical challenges
that haven't been properly addressed?

[01:11:21] More labs, I think, is still interesting.

[01:11:24] Just can we do more on model
architecture is always interesting.

[01:11:27] You think we have not fully
explored model architecture space?

[01:11:30] The Frontier Labs have done a pretty good
job of exploring it, but I think,

[01:11:36] as the hardware changes, the shape
of the model should change for sure.

[01:11:41] Presumably you're not thinking like,
yet another Frontier Lab

[01:11:44] pursuing the same architecture.

[01:11:45] You think there's probably off the wall
looking architectures that will actually

[01:11:49] make a lot of— I think there's
a little bit off the wall for sure.

[01:11:53] Do you have a specific
architecture in mind?

[01:11:55] My mentality is always sticking within
the transformer family, but

[01:12:00] what are the constraints that are
currently available,

[01:12:02] currently imposed that you could lift?

[01:12:05] For example, one of the things is there's
this idea, when you're doing transformer

[01:12:10] inference, you do
pre-fill that is processing what the user

[01:12:13] said to you, and then there's decode
which is generating the response to that.

[01:12:17] Those are totally different in pretty much
every aspect of how they actually run.

[01:12:22] One runs a step at a time,
the other one runs really in parallel.

[01:12:26] There is this somewhat artificial
constraint today that those are

[01:12:29] the same model that's doing both.

[01:12:31] Maybe lift that constraint.

[01:12:33] Another example would be,
there's this idea that the model that you…

[01:12:37] This is more fundamental constraint
that you have to train

[01:12:40] the same model as you serve.

[01:12:42] But again, training is very
different from serving.

[01:12:45] At training, it's very compute intensive.

[01:12:47] At serving, it's more
memory bandwidth intensive.

[01:12:50] Maybe, is there a way you can make a model
that when you use it at inference time, it

[01:12:56] increases the amount of computing it does
to use some of the available resources?

[01:12:59] Makes sense.
Reiner, thank you.

[01:13:01] Pleasure.
