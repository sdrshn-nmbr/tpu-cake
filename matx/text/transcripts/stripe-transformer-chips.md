# Reiner Pope of MatX on accelerating AI with transformer-optimized chips

[![Cheeky Pint's avatar](https://substackcdn.com/image/fetch/$s_!wNfQ!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fadc7c81d-031d-45d2-9c29-e315f8586925_1821x1821.png)](https://substack.com/@cheekypint)

[Cheeky Pint](https://substack.com/@cheekypint)

Feb 26, 2026

3

1

Share

Reiner Pope is the co-founder and CEO of MatX, designing specialized chips for Large Language Models. A former Google TPU architect, he joins John to discuss why the current generation of AI hardware is hitting a wall. They cover the "uncomfortable trade-off" between latency and throughput for current chips, why MatX is betting on combining HBM and SRAM to solve it, and the massive logistical challenge of manufacturing chips at scale with TSMC. Reiner also shares his predictions for AI in 2027, why he prefers Rust for hardware design, and why the best iteration loops happen in your head before writing a line of code.

#### **Timestamps**

00:00:15 Google’s AI revival  
00:07:54 MatX  
00:17:11 AI supply chain  
00:21:48 Designing chips  
00:37:11 TSMC  
00:44:17 Token pricing  
00:44:55 RL-ing chip design  
00:49:26 Design to production  
00:56:05 MatX culture  
01:02:57 Rust  
01:05:21 Cuckoo hashing  
01:09:35 Unexplored model architectures

Thanks for reading! Subscribe for free to receive new posts and support my work.

Subscribe

#### **Transcript**

[00:00:01.07] John Collison

Reiner Pope is the co-founder and CEO of MatX. He’s a former math whiz and Haskell programmer who became a TPU architect for Google. Now he’s teamed up with Google’s former chief chip architect to design a better chip for AI.

[00:00:16.04] John Collison

A year ago, everyone was saying Google is canceled. AI is going to eat their search. No one’s going to search for things, and therefore, the business won’t do well. Obviously, that sentiment has really shifted, in part helped by Gemini 3 is really good. Then also it’s really fast. It’s powered by the custom chip hardware Google has. You were inside Google for a lot of the foundational period, laying the groundwork for that stuff. What do people not appreciate about what Google did right to lay all the groundwork for their current AI success?

[00:00:52.15] Reiner Pope

They started with the research. The transformers came from there. Pretty much anyone who’s over 30 and at a large lab has been at Google Brain at some point. I think there was and has been a lot of talent there. TPUs are pretty good. We think there’s better you can do, of course, but they at least had the option, the opportunity to design the TPUs for neural nets at least, rather than graphics applications like NVIDIA. The overall architecture, starting with single core doing what was at the time reasonably large systolic arrays, by today’s standards, nowhere near as much, but I think those were a lot of really good decisions.

[00:01:35.00] John Collison

When did the TPU project start?

[00:01:37.04] Reiner Pope

TPU v1 was announced in 2016, I think. That was what led to the creation of all of those 2016–2017 startups. Cerebras, Groq, Graphcore, SambaNova, all of those. TPU v1 is a really impressive project. It was done on a very short timeline. I don’t know the full details, but maybe about a year or so, maybe a year and a half, with a skeleton team of 20-30 people. Really, really minimal viable product. More recent TPUs and more recent AI chips in general can’t do that because the market has moved, and the table stakes are much higher. The first-generation product, they just… One big systolic array, stick a memory next to it, we’re done. It was really simple, nice, elegant product.

[00:02:26.16] John Collison

Obviously, TPU v1 predates the Transformer. Is that just a coincidence that they happened at very similar times, or are they related in some way?

[00:02:36.01] Reiner Pope

There was a period of maybe about four years of a lot of ML research, or neural net research, prior to Transformer. What was popular? LSTMs and ConvNets and ResNet and Inception. The big thinking at the time was to adapt it to be used for LSTMs. It’s a reasonable fit there. But I think there was just a huge flurry of activity. Why did it all happen then and not later? It’s probably just because people stopped publishing. 2022 was about the time when Google completely stopped publishing its research. All the good papers are from before that as a result.

[00:03:21.19] John Collison

But is there some hand-wavy story you can tell about parallelization where both Transformers and TPUs are about really internalizing the importance of parallelization?

[00:03:36.12] Reiner Pope

Definitely, I put it somewhat on people, actually. It is just true. Hardware is massively parallel. You’ve got tens of billions, hundreds of billions of transistors on your chip, and it takes maybe 100 clock cycles to get from one side of the chip to the other. You can’t do a sequential computation involving transistors on both sides of the chip. The hardware is just fundamentally parallel, and you have to take advantage of that.

[00:04:03.02] Reiner Pope

TPU v1 and all later TPUs naturally took advantage of that. Matrix Multiply is really nice because it is so parallel. I think on the hardware side that’s generally understood. I think most ML researchers, especially of the time, were not super deep in what hardware wants, and what is… Mechanical sympathy is sometimes a term that’s used for that.

[00:04:28.09] John Collison

What is the term? It kind of—

[00:04:30.14] Reiner Pope

It speaks for itself. Think about the poor machine and what does it want?

[00:04:37.23] John Collison

What does it want?

[00:04:39.02] Reiner Pope

The term actually, I think, originates in maybe high-frequency trading in areas like that, which is, I haven’t worked in. I’ve just, I like reading about the software that people have built from there. It’s like, for them, what does the machine want? It wants a lot of instruction-level parallelism. This is CPUs, not TPUs. Once a lot don’t branch, so unpredictable branches kill your performance. Think about the things that CPUs do and how to use them best. Can I get to peak performance on a CPU? It’s that idea.

[00:05:10.14] Reiner Pope

I think the whole idea of peak performance on a CPU is kind of crazy. No one even says, “What is peak performance? What is my percentage of peak on a CPU?” Because the performance of software running on CPUs is really bad, but running on GPUs, or TPUs, or AI chips in general, actually that is the main focus. It’s like, “What is my percentage of peak? Can I get 70% or 80%?”

[00:05:30.03] John Collison

I feel like many people listening to this know that GPUs perform better for AI workloads than CPUs. It’s a funny history when you think about it, where just one day we woke up with all these very mathematically intensive workloads, first crypto mining, and then AI, so then NVIDIA is extremely well-positioned because they’ve been making GPUs for gamers that you would plug into... You’d buy your Dell PC back in the day and maybe upgrade the graphics card by plugging in a better NVIDIA graphics card than the one the stock Dell computer came with.

[00:06:10.11] John Collison

They were incredibly well-positioned to capture that. I think people know that. What is the intuitive explanation as to why GPUs are better for AI workloads than CPUs? People say, “They’re better for these mathematical computations.” But that’s a tautological answer, basically. Is there some way you can have a mental model for why that is the case? Because software instruction sets also involve doing math.

[00:06:35.20] Reiner Pope

Intuitions, I’m not sure. Let me try and just go to some of the big differences, which is, really wide vector instructions is the hallmark of a GPU, which I think it’s maybe… If you want some intuition, it’s like how much is spent on controlling the thing? Maybe control means, if I’m driving a truck, how much… Is the driver versus the payload?

[00:07:04.04] Reiner Pope

A truck has a huge payload in it. That’s more like the GPU, whereas maybe a motorcycle is more like the CPU, where you’ve got… Actually just processing the instructions, reading, “What do I have to do next? How do I do that?” That is most of the cost on a CPU, whereas if you just keep the same instructions but make the payload 100 times bigger, then you can shift most of the cost to be in the actual work that you want to do.

[00:07:29.14] John Collison

CPUs have been optimized for very complex instruction sets, whereas GPU is optimized for—

[00:07:38.12] Reiner Pope

Complex instruction sets and fine-grained changing what you want to do. Like steering, in this analogy, a CPU can steer an obstacle course, no problem, whereas, on a GPU, you’re just going to go straight line for a really long time.

[00:07:54.02] John Collison

This is getting us into... What is MatX? How did you guys start it, and which part of this space are you attacking?

[00:08:03.04] Reiner Pope

MatX is making the best chips physically possible for LLMs. What led us into MatX… Mike is the other founder. Mike and I were both working at Google. I was working on the inference stack for running LLMs, and I was saying, “How can we make the best software on TPUs for running LLMs?”

[00:08:33.17] Reiner Pope

Then what we really wanted out of hardware was support much, much larger matrices. The matrices have grown from maybe 128 in dimension into the many thousands. Truck goes to many trailers. Much larger matrices and much lower precision arithmetic. We tried to move the TPUs in this direction. TPUs have been moving in this direction, but they’re constrained by a lot of other workloads. There was a big ads workload at the time.

[00:09:06.19] Reiner Pope

Back in ‘22, before ChatGPT was released, there was this idea that LLMs were going to be a big thing, but not conviction, and really hard to make a big bet on that. I think a startup is more of the right place to make a big bet on a workload. If you fail, it’s fine. Another startup will succeed. Whereas I think a company like Google or NVIDIA, the next chip has to work for sure.

[00:09:34.04] John Collison

You can take more technical risks as they turn up.

[00:09:37.03] Reiner Pope

Actually, I would say we were taking product risks rather than technical risks.

[00:09:41.05] John Collison

But is there actually product risk? Because it seems like LLMs are going to work.

[00:09:45.18] Reiner Pope

I think now we understand it. Two or three years ago, I think it was just like a—

[00:09:50.15] John Collison

And when you say the best chips for LLMs... I can think of multiple ways to measure best. It could be the best performance per watt. It could be the lowest latency, capable of handling the largest models. What is best?

[00:09:59.17] Reiner Pope

In general, there are two metrics which LLM workloads care about, which is throughput, which is really just an economics thing. I buy a chip for $30,000, and then can I do 10,000 tokens a second or 100,000 tokens per second of throughput? That determines the dollars per token. Throughput and then latency, how fast does a thing respond? As I see the market, the economics seems to be most important.

[00:10:28.07] Reiner Pope

Ultimately, the quality of the AI you can train and serve is constrained by, “I have only a $10 billion budget, and I want to train and serve the best model I can on that budget.” If I can have more tokens per dollar, then I can get a better quality out. The product we aim to build is far ahead on throughput, but then, actually, the surprising thing is we’re competitive with the best on latency as well. I think that is a unique thing in offering both in the same place.

[00:11:00.21] John Collison

Is this for… Obviously in AI, there’s training the models and then running the models’ inference. Is this most interesting for inference, or is there any training angle? Incidentally, is it useful for trading, but you are trying to win inference, is that how you think about it?

[00:11:15.17] Reiner Pope

I think that’s a reasonable way to look at it. I think the best inference chip today will be a really good training chip as well. Our product is both training and inference, but I think the first sales will be an inference. That’s mostly just a market effect where it’s easier to buy, it’s not as big of a risk to go to buy an inference cluster as a training cluster. I think the product is really compelling for training as well. I think it should be the best training product.

[00:11:42.20] John Collison

You guys just raised a big new round of financing.

[00:11:45.21] Reiner Pope

That’s right. We’ve raised a series B round, it’s led by Jane Street and Situational Awareness. Situational Awareness, that is, Leopold Aschenbrenner’s fund. He wrote the definitive book on AGI and where it’s going. Then Jane Street, they’re real technical experts. They understand all the details really well. Very happy to be having them lead the round. It’s a $500 million round, helps us actually ramp the manufacturing and supply chain for our chip so we can bring our chip to market.

[00:12:21.17] John Collison

That’s a lot of money.

[00:12:22.12] Reiner Pope

It is. I think roughly, I would say it costs a ballpark $100 million to produce a chip in small volumes. But then you see the orders that are going around, like OpenAI, Anthropic, Google. They’re going around buying multi-gigawatt clusters, they cost tens of billions of dollars of chips, and you want to deploy all of that in a year or so. You just need a massive supply chain behind you.

[00:12:49.10] John Collison

Assuming everything works technically, what rate of production could you start to see?

[00:12:57.14] Reiner Pope

We have some estimates of where we would like to be on this. Ramping to very large volumes is a huge challenge for anyone. Obviously, for the large players, they’ve had some practice at it. Getting to a very large volume for a startup is hard. We would like to be at a place where we’re shipping multiple gigawatts a year.

[00:13:16.09] John Collison

Multiple gigawatts per year?

[00:13:18.20] Reiner Pope

Yeah.

[00:13:18.23] John Collison

Speaking of the metrics, you talked about tokens per second. We used to measure chips in FLOPS, and I guess there’s some kind of custom FLOP thing for AI chips. But is everyone just using tokens per second these days? Is the industry aligning on that as the chip metric?

[00:13:31.22] Reiner Pope

I guess it’s like an application metric versus the chip itself. FLOPS of the chip is the key chip metric. There’s a little bit of, if I go and say, “I’ve got an exaFLOP chip,” to you then the appropriate suspicion is to say, “But can I actually use those FLOPS effectively?” Then you need to map the application to that.

[00:13:56.06] John Collison

This is kind of telling you the usable FLOPS for your purposes. As a consumer of AI, we have known for a long time that lower latency products succeed. Google talked about their internal testing where the differences were down to… Was it 50 milliseconds?

[00:14:18.07] Reiner Pope

Something like that.

[00:14:18.20] John Collison

In result times where they noticed more Google engagement, the faster the results were, and you’d think that 50 milliseconds is imperceptible to a human. It almost is, but turns out it’s not. I think Amazon has, certainly, they’ve optimized the latency of the Amazon experience quite a lot. I don’t know if they’ve talked about this stuff publicly, but you know that their internal metrics similarly show that the faster the product page loads, the more people buy it.

[00:14:44.01] John Collison

Yet in AI, Google has carved out a meaningful advantage via Gemini just being really fast for its level of intelligence. As far as I can tell, ahead of most of the other labs at a latency, at a fixed high level of intelligence. Why have you guys or Groq or better chips not been adopted faster to give this product latency? It’s just that this will happen and you guys will be powering all the AI products, but I note that Google has an interesting lead there.

[00:15:22.12] Reiner Pope

I think there’s ultimately… At least for existing chips in the market, there’s a really uncomfortable trade-off between latency and throughput. The chips that are best at throughput have historically been the chips that are based on HBM as the memory. That is Google, Amazon, NVIDIA.

[00:15:39.06] Reiner Pope

In order to have very large throughput, you need a lot of inferences in flight simultaneously. That needs a large memory. But that hasn’t been so good at latency. Then, there’s the Groq and Cerebras that are much better at latency because they’ve got this, the SRAM, weights are in SRAM, very low latency.

[00:15:56.02] Reiner Pope

The problem is, and the challenge when you go to a Groq or Cerebras system is that the throughput you get there, it just is not very good. The fundamental dollars per token is just not competitive with Google or NVIDIA or Amazon. It is actually possible to do both in the same chip. It’s kind of an obvious thing. You say you take the HBM, you take the SRAM, put them together on the same chip, you put the weights in SRAM, and you put all of the inference data in HBM.

[00:16:24.22] Reiner Pope

That is what we are doing, in fact. I think that actually hits a really nice sweet spot where you can get low latency and also be very cheap. I think that’s a really attractive point to be. It hasn’t happened in the market yet, just because of product decisions that have been made by the different chips.

[00:16:38.12] John Collison

But we should expect all the AIs we’re using to get significantly faster over the coming 3–5 years?

[00:16:45.04] Reiner Pope

Order of magnitude faster, I’d say. Generally, HBM-based chips tend to be about 10 milliseconds or 20 milliseconds per—

[00:16:52.19] John Collison

I’m sorry, HBM-based chips are things like TPUs?

[00:16:55.05] Reiner Pope

That’s right. There’s just some simple math of, how long does it take you to read through all of HBM? It takes about 20 milliseconds. That’s the amount of time per token it runs, whereas the amount of time to read through all of SRAM is much faster. You get about 1 millisecond, so they are managed pretty faster.

[00:17:11.09] John Collison

Famously, software used to be… Old-fashioned deterministic software, the kind that’s now out of favor, used to be very easy and quick to scale. You had a social networks that have some Southwest, Northwest moment, and they can scale through 10, 100, 1,000 orders of magnitude of adding users because it’s just a few rows in a database, and it’s a very underutilized CPU. What’s interesting about the AI world is there are very real bottlenecks. Elon spent lots of time talking about power, but it’s not just bringing power online. You mentioned HBM is reminding me of… It seems like there’s a view that maybe there’s going to be some HBM supply chain crunch. Where do you see…

[00:18:01.13] John Collison

Are we in for just a crunched world where some limiter is pacing the rate of AI buildout over the coming few years, where the economics work, and the products and everything like that, but ultimately, we just can’t bring the components online fast enough because we have to build out the factories, things like that. What are those crunched components?

[00:18:23.02] Reiner Pope

I think so. I’ll just comment, by the way, this is a great time to be a supplier in this place. Or just really—

[00:18:30.18] John Collison

You should have started an HBM company.

[00:18:31.20] Reiner Pope

I know. I think it’s also just a fun time to be someone who optimizes software. That’s always what I like doing. Always the challenge is, “Why am I optimizing this if no one cares?” But finally, there’s a place where you can… It’s actually very meaningful in a very tangible sense. If I can make this 20% more efficient, then it can save that 20% of the buildout.

[00:18:54.12] Reiner Pope

The supply chain, we’re going to have crunches on all of the supply chain really. If you look at the big components of what any company, like us for example, build out, there is dependency on logic ties from typically TSMC, maybe Samsung, or HBM, which are the big three HBM vendors. Hynix, Samsung, and Micron. Then there’s also just the whole rack manufacturing, which includes literally just sheet metal and so on that builds the rack, but also cables and connectors because of all the high-speed interconnect. That’s what we—

[00:19:30.01] John Collison

Racks don’t sound hard. Are they sneaky hard?

[00:19:33.05] Reiner Pope

The big challenge is that you want to bring in a huge amount of power, get a huge amount of heat out, and also have phenomenal interconnect, which has very high signal integrity requirements. Pack a lot of cables in with… The cables don’t bend too much, they have to have enough copper in them, and so on, that you don’t lose data rate on the interconnect. If you push it to a limit, it’s sound.

[00:19:55.17] John Collison

Wafers, racks, HBM. What else?

[00:19:59.15] Reiner Pope

Data centers, which I think is power, primarily a little bit of buildout, but primarily power and good infrastructure there.

[00:20:07.04] John Collison

How do you then, as a startup that is looking to acquire all these components, elbow your way in amongst the giants of the Googles and the NVIDIAs and all these people who have long-running relationships and have been buying for much longer?

[00:20:25.12] Reiner Pope

Ultimately what all of these suppliers care about, they do somewhat care about a diversity of their own customers. It’s not a great position to be in.

[00:20:35.20] John Collison

They don’t want monotony.

[00:20:36.21] Reiner Pope

That’s right, yeah. But then, what is their hesitation? Or the calculus for one of these large suppliers is, if I reserve some of my capacity for you, a startup, are you going to be around in a year? Is anyone going to even buy your product? Our approach has been to just actually find buyers for the product, and then the buyers answer that question, ultimately.

[00:21:03.04] John Collison

If you show up with a bunch of fairly ironclad contracts to a supplier, then that has helped.

[00:21:09.23] Reiner Pope

That’s the nature of it, yeah.

[00:21:11.01] John Collison

I presume also the round you just raised really helps there, where showing that you are incredibly well-capitalized and not going anywhere also helps from a supplier validation point of view.

[00:21:24.17] Reiner Pope

Absolutely. It helps just to say that we are around. We, in some cases, are actually… It depends on which part of the supply chain, but some parts of the supply chain, some are fungible. Logic ties are typically pretty fungible. But other parts of manufacturing, you actually need something specifically set up for you. We’re also able to cover the capital costs of that.

[00:21:46.10] John Collison

That makes sense. Coming back to the MatX architecture. You want to build the best chip for LLMs.

[00:21:54.00] Reiner Pope

What is that?

[00:21:54.15] John Collison

Yeah, exactly. Sounds great.

[00:21:57.17] Reiner Pope

There’s a few aspects to that. I think the first one is pick your memory system right. I said, we’ve seen this HBM family, we’ve got the SRAM family, put them both together is actually, the most obvious idea, but you can actually do it. There are a lot of details to make that work well. We’ve done that work. One of the things that shows up there is you’ve spent all of this area on your chip on SRAM. How do you fit in the matrix multipliers, which are the other big thing you really need to do, and somehow create a much more efficient matrix multiply engine? There is a gold standard for that. That is called the systolic array. Make a really large systolic array. You can’t beat that in area or power efficiency.

[00:22:37.11] John Collison

Provably so? Practically?

[00:22:39.14] Reiner Pope

Practically. It has not known a better approach there. The main thing is, where are the inefficiencies typically? The inefficiencies show up when you leave the systolic array. If you make a systolic array really big, then you just don’t leave it as often. That’s the idea. Make a really big systolic array. That is sort of the theme of several of the 2023-era startups, including us. But one of the challenges there is, now, there is this part of the neural network, as part of the Transformer, which is this attention that doesn’t map well onto a large systolic array. That’s the tension.

[00:23:17.23] Reiner Pope

The mixture of expert layer maps really well, but the attention does not. What we came up with, which is quite different than some of the other startups in this space, is say, take a really large historical array, but have a way to split it up into pieces without losing efficiency. That is the core of the design for us.

[00:23:38.07] Reiner Pope

Then, the third component. First was HBM and SRAM, second is the systolic array, third component is an interesting new approach on low-precision arithmetic. Low-precision arithmetic, in general, we’ve seen number formats get narrower and narrower. They get faster and faster as you make them less precise.

[00:23:57.06] John Collison

Number formats get narrower. What does that mean?

[00:23:59.03] Reiner Pope

Float32 was how people used to train neural nets.

[00:24:04.14] John Collison

That’s just too much precision. It’s wasteful.

[00:24:06.06] Reiner Pope

Too much precision, yeah. It’s like saying, “I’ve got an image with a billion color bit depth.” It’s too many colors. You’d rather have more pixels and fewer colors. That trend seems to go almost all the way down to one bit even, where just have very few colors but a huge number of pixels. In net seems to be a better, just more efficient way to train models.

[00:24:35.08] John Collison

Sorry, literally what precision are you dealing with in this sense?

[00:24:42.07] Reiner Pope

We have a range. We actually have an ML team who we hired specifically to research different forms of numerics and how to make them all work together really well. We have a range of precisions. It’s not just one precision. We think probably the main thing will be similar to where NVIDIA is at, which is 4-bit precision. But I think a mix of different precisions is useful for just when you look at the research, sometimes you want some layers in higher precision or lower precision, and so on.

[00:25:13.23] John Collison

4-bit is 16.

[00:25:15.19] Reiner Pope

Yeah, you get 16 choices. That’s it.

[00:25:17.06] John Collison

That’s it. It’s pretty imprecise. That’s really interesting. I didn’t know about that dynamic, but it makes sense.

[00:25:23.18] Reiner Pope

Half of them are positive, half of them are negative. It’s even—

[00:25:29.12] John Collison

How do you design a chip? Is that a whiteboard? What software are you working in? I’d just love to know… I understand how you design software and what that process looks like. I have actually no sense for what chip design looks like.

[00:25:42.16] Reiner Pope

The way that you actually type a chip into a computer is similar to software. You write Verilog. Verilog is a programming language. It is a very parallel programming language, which makes it different than C or Python or something. But it is a programming language. The mechanics of how you express the design are the same as software, and we have continuous integration, Git, all of those things.

[00:26:05.02] John Collison

But a program executes... Like your Verilog program...

[00:26:10.05] Reiner Pope

We don’t really run it.

[00:26:11.09] John Collison

Exactly. How does it run?

[00:26:13.14] Reiner Pope

We synthesize it. Synopsys and Cadence provide EDA tools.

[00:26:18.07] John Collison

EDA? You have to remember, I’m just—

[00:26:19.13] Reiner Pope

I don’t even know what it means, really. I think it’s Electronic Design Automation. It takes the Verilog and says… First turns it into a description of what are the logic gates that are involved, ANDs, ORs, NOTs, and then the wires between them. Then it runs for days doing some really difficult algorithms, and then eventually produces… Gates are the first thing, and then even below that, it literally just produces polygons. It says like, P-type semiconductor here, N-type semiconductor here, and polysilicon.

[00:26:55.11] John Collison

You write Verilog, and then that compiles down into gates and ultimately the Minecraft 3D. “This is where your elements should go.” But then, what is the iteration loop? When we write code at Stripe, we build a first version of something, and then we try it out, and then we refine it, and we add more functionality over time. We’re going to write some tests at some point. We’ll ship that, we’ll find product market fit, and then we’ll refine it in market. Do you just sit down and write the completed chip, and it works really well?

[00:27:35.04] Reiner Pope

Every year we tape-out a chip, and if there’s a bug, we just wait till next year. It’s not really how we do it.

[00:27:39.20] John Collison

What’s the iterative process?

[00:27:41.04] Reiner Pope

How do we actually do it? It’s much more waterfall than software is. Waterfall is almost a bad word in software development.

[00:27:47.07] John Collison

But it’s a fact of life in general.

[00:27:48.23] Reiner Pope

The waterfall goes from architects to logic designers who are writing Verilog. Then, there’s this design verification, and then physical design. There’s this really big architecture phase which happens before even writing any Verilog, which is, “What do I want the organization of my chip to be?”

[00:28:12.23] Reiner Pope

In some sense… I came to hardware after doing almost 10 years in software. I really like the blank slate you get in hardware. You’ve got all of the raw materials, you have a much more varied in what you have available. What is the organization of your chip? Do I have 100 cores? Do I have one core? Do I have systolic arrays? Do I have vector units? All of those things.

[00:28:36.02] Reiner Pope

Then we spend a long time coming up with that general principle and then saying, “Okay, now I’ve got these applications I want to run. I want to run a transformer of a particular shape. I want to map that onto this architecture that I’ve got in my head.” We do a lot of iteration. Well, I’ve got this architecture in my head. I write it down to communicate to other people, but that’s just like a markdown file. Then, still actually a lot in my head, but maybe with Python simulation and so on, I’ll see, do my applications map well to it? Can I run LLMs?

[00:29:06.17] John Collison

You have a simulator where you write your chip, you can then simulate its performance, and you have some battery of tests that you see how this chip design works. Is it like an industry standard... Is it the X-Plane of chip testing?

[00:29:25.05] Reiner Pope

There’s an industry standard thing for the Verilog once you’ve done the design. They’re just Verilog simulators that you can test against. But you’ve already invested a huge amount of work by the time you’ve got to that point, and so you sure hope you haven’t made a big mistake at that point. The thing that everyone does prior to that is, we’ll write our own performance simulator, which, I mean, it is very specific to your particular architecture, and you can write it quite concisely in just a normal programming language. That is where most of the architecture work is done. Then the simulation on Verilog is more, “I know what I’m doing. I just want to make sure I didn’t have any bugs when I implemented it.”

[00:30:06.10] John Collison

But I presume it’s a game of inches where different people are trying different things, and then you... Do you simulate it to see if it runs 1% better across the battery of tests? Or is that not how it works?

[00:30:19.21] Reiner Pope

In this space, not so much. Just to characterize what performance of an AI chip is, it is how many, really… First thing you care about is FLOPS. How many FLOPS have I got? That’s a product of how many multiplies, like, “I’ve got a grid of a certain size, 1000 by 1000.” Can do a million multiplies in a clock cycle. Then I have a certain clock frequency, a gigahertz. I multiply them out. That is the speed of it. I don’t even need to write that and test it to see how fast it is.

[00:30:50.14] John Collison

It just is.

[00:30:51.13] Reiner Pope

What I plan in advance is it’s going to be this fast. What I can then optimize on, maybe a little bit, is clock speed. There’s not a lot I can do there. Then, I can optimize a bit on area as well. There is some room for optimization, but actually a lot of it gets set. Actually, just the speed of the chip gets set very much upfront.

[00:31:10.02] John Collison

Then how many chips do you fab? Is it only the ones going into production, or is it just build a few to throw away, or how does it work?

[00:31:20.02] Reiner Pope

The ideal, which companies tend to hit about 50% of the time, is that your first tape-out… Tape-out costs $30 million. Your first—

[00:31:29.10] John Collison

Tape-out is just production run?

[00:31:30.19] Reiner Pope

That’s right. The actual manual, the first chip costs $30 million, the second chip costs $1,000.

[00:31:36.23] John Collison

Yes.

[00:31:38.08] Reiner Pope

Tape-out is that first chip.

[00:31:39.23] John Collison

Okay.

[00:31:40.14] Reiner Pope

The ideal is that your first tape-out is actually your production thing. You do a tape-out, you make maybe a thousand chips and test them, and then you do production volume. In the unlucky 50% of the time, you need to redo some or all of your tape-out. In good cases, and in many cases, you can redo just the metal layers which costs you only like $100,000.

[00:32:05.05] John Collison

As opposed to the—

[00:32:06.20] Reiner Pope

Pay the $30 million again. But in bad cases, if you’ve made something serious and you can’t fix it at the metal layers, you have to do the whole thing again.

[00:32:14.18] John Collison

Why can’t that be solved? Is that definitionally an error in simulation, where it turns out these two gates were too close together, and it just led to some reliability issues?

[00:32:27.18] Reiner Pope

Yeah. What you’re describing is physical, the physical implementation of the chip is wrong. That’s one class. The other class is that the logical specification of the chip is wrong.

[00:32:41.13] John Collison

But shouldn’t that be—

[00:32:42.23] Reiner Pope

Shouldn’t you have caught that before? Yeah.

[00:32:44.20] John Collison

Before you spent $30 million on manufacturing it.

[00:32:46.22] Reiner Pope

Yeah. We do a lot of testing. We try not to ship these things. I hear software companies also ship bugs to production as well.

[00:32:56.01] John Collison

Fair.

[00:32:56.05] Reiner Pope

Sometimes things—

[00:32:58.20] John Collison

It’s a very good retort. Shouldn’t you not be shipping bugs?

[00:33:03.15] Reiner Pope

But there is a real trade-off in, you can spend more and more time on design verification. There’s always this question of, when do you stop? You stop when your coverage metrics ever hit a certain point, but maybe not 100%.

[00:33:18.21] John Collison

Then if Apple has to discretize the iPhone release cycle, and they have settled on once per year, they’ll decide, “We’ve got this better camera, but it’s got to wait for the next version,” or, “We’re going to improve the waterproofing, but that’s got to wait for the iPhone 8 or whatever.” They have taken a continuous process of always coming up with ways to make the iPhone better and discretized it into annual iPhone releases. What will your discrete cadence be?

[00:33:51.15] Reiner Pope

What’s our vision of that?

[00:33:52.02] John Collison

Yeah.

[00:33:52.12] Reiner Pope

Many chip vendors have this sort of tick-tock model, which is you’ll do on one generation, maybe you’re trying to release every year. On even numbered years, you’ll do a physical technology upgrade, so new transistor technology, new memory technology, and you interconnect. Then on odd numbered years, you might do an architecture overhaul. I think that’s a pretty good fit because you have different parts of your company that are skilled at different areas, and it allows you to keep both of them occupied without having instead every 2 years doing a massive risk release.

[00:34:25.14] John Collison

Yeah. So you think that’s probably likely for you?

[00:34:27.20] Reiner Pope

That’s right.

[00:34:29.23] John Collison

You mentioned interconnect. There’s a narrative out there that in NVIDIA, a huge part of the defensibility comes not from the chips, which are good, but from the software layer, and the ability for engineers to write these really parallel workloads, and the fact that they’ve been refining CUDA for whatever number of years.

[00:34:48.12] Reiner Pope

A decade or so.

[00:34:49.06] John Collison

Exactly, a long time. How do you think about parallelization, and is that narrative true?

[00:34:56.09] Reiner Pope

It’s true for sure. It’s true in many areas of the market. I think, and especially where you look at where NVIDIA entered the market, they’re doing PC devices, lots of gaming, and so on. There are thousands of games, maybe tens of thousands of games released, and they all need to be programmed against CUDA, and so there’s such a huge investment in the software that this is really important, the compatibility.

[00:35:28.03] Reiner Pope

There are not thousands of LLMs. There’s one LLM per frontier lab, and there’s maybe five frontier labs or something like that. Just the economics of that is different. The calculation for a frontier lab roughly goes as, I just bought a $10 billion compute cluster, I have hired 50 of the best people who can write optimized GPU, or TPU, or Trainium software. I pay them less than $10 billion, a lot less. Let’s put them to work optimizing the compute.

[00:36:07.11] Reiner Pope

Good work there, depends on what your baseline is, but it can very easily double the performance of the software you write. There is a huge amount of custom software written for every generation of chip. When a new chip comes out, software is substantially rewritten to optimize for that specific chip. That’s just the right trade-off given the relative costs of these things. What that means for us is that ecosystem already exists, and that way of operating, where you say, “I’m just going to staff a 50-person team to write software for this chip,” works really well if you’re trying to sell to frontier labs.

[00:36:44.03] John Collison

You’re saying CUDA is way more important for the games environment, where it just does a lot of games than this top-heavy AI market that we’re in, where if people say, “You need to then customize your workload for a MatX chip,” it’s like, “Well, fine, do that.”

[00:37:06.01] Reiner Pope

Cost of business.

[00:37:06.17] John Collison

Yeah, that makes a lot of sense. Where will you fab the chips?

[00:37:12.18] Reiner Pope

TSMC.

[00:37:13.15] John Collison

Why is TSMC so durable?

[00:37:21.16] Reiner Pope

It’s interesting. They don’t charge a lot as well. You’d think that if they’re a monopoly provider, they should charge a lot of money. They don’t. I think that is a big aspect of why they’re so durable.

[00:37:32.11] John Collison

It’s like this cyclical conservatism crossed with Taiwanese business conservatism, means you’re at the most conservative part of the matrix.

[00:37:43.22] Reiner Pope

But I mean, it does… I mean, an American capitalist might say, “Well, they’re just screwing up. They could have extracted more money from the market.” But you could also say that this is actually the long-term sustaining advantage because they will just stay ahead for a really long time.

[00:38:01.03] John Collison

They don’t incur the creation of competitors. But isn’t the creation of competitors priced in because of the geopolitical risk? It’s not like everyone’s fat, dumb, and happy with their TSMC dependence. They’re actually thinking a lot about it.

[00:38:14.22] Reiner Pope

So there is real technical advantage there as well. It’s not just the discouragement.

[00:38:19.07] John Collison

But designing chips seems really hard, building airplanes seems really hard. There are so many areas where competitive market forces create multiple options. Yet, that has not occurred here.

[00:38:33.13] Reiner Pope

There are multiple options. You can buy from Intel or Samsung.

[00:38:37.17] John Collison

But at leading-edge nodes.

[00:38:38.23] Reiner Pope

What do we even care about in leading-edge nodes, I guess? The big advantage is on power. The advantage on area is smaller. The leading edge nodes, the density doesn’t go up as much as it used to. When you are really, really sensitive to power, it is a good idea to be on leading edge nodes. That is AI chips and mobile phone chips. But there’s a lot of the market where you don’t actually like the devices in your car.

[00:39:03.04] John Collison

Sure. Car chips, yeah, that’s fine. But you’re saying, if you exclude the two most interesting parts of the market—

[00:39:16.07] Reiner Pope

That’s true.

[00:39:16.14] John Collison

For this super high-growth area of the market, it’s interesting to me, again, there’s a lot of other really complex business problems out there that competition has solved. Chip design is a… Like, why has someone not left TSMC and gone and built a new fab?

[00:39:28.21] Reiner Pope

I don’t know. The cost of a fab is extremely expensive. I recognize that also the cost of a lab is extremely expensive, too. I don’t really understand the technical details of why it’s so hard. I mean, there is some amount of just a $10 billion fab versus $100 million tape-out and chip development. There’s a huge difference there. But beyond that, I’m not sure.

[00:39:55.08] John Collison

What’s TSMC like to deal with?

[00:39:57.01] Reiner Pope

They’re very big. As a startup, we tend to work with, not directly with TSMC, but with an ASIC vendor who, firstly, does a huge amount of the actual backend work for us to interface with them, but then also has their existing relationships with them. TSMC cares a lot about diversity of their customer pool.

[00:40:18.06] John Collison

It gets back to that conservatism.

[00:40:19.16] Reiner Pope

Yeah. They’re great to work with from that perspective.

[00:40:24.04] John Collison

They want to encourage startups.

[00:40:25.04] Reiner Pope

That’s right.

[00:40:26.06] John Collison

That’s very cool. Why don’t the labs design their own chips? Google does.

[00:40:30.03] Reiner Pope

Google does. OpenAI is starting. It’s really a trade-off of how much advantage do you get from vertical integration versus how much advantage do you get by concentration of R&D work. You take the five labs, and if they all buy from one player, then you can put five times as much R&D into that chip. Does that beat the advantage you get from saying, “I know exactly what my model is”?

[00:40:52.18] Reiner Pope

Because of the several-years delay from designing a chip to being in production, you can’t actually say, “I know exactly what my model is,” because models change much faster than that. Even the labs are forced into this position where they have to make predictions, and they have to hedge against what they might do two years from now. The calculus is, what is the probability distribution of what my model might look like and then design a chip that gets 90% of that probability distribution or something?

[00:41:22.00] John Collison

Elon is excited about data centers in space. The two criticisms I’ve heard are that cooling is very hard and then repairing the chips is hard. But I know nothing about chips. You do.

[00:41:39.17] Reiner Pope

The repair is really interesting. When you look at how NVIDIA deploys their racks, how we do something pretty similar to what NVIDIA does. In general, you always need to design for the fact that some of your chips are going to be down. Mean time between failure of chips is not that large. In a cluster of 100,000 chips, there are going to be chips that are down all the time.

[00:42:02.12] Reiner Pope

One way you can do that is you can make a rack where one rack has spare chips in it. NVIDIA has eight spare chips in a rack of 64. That’s pretty good. The combinatorics works really well for you there. Because you can pick which ones to avoid, you can with very high probability tolerate a lot of failures. And then the other just for the other family of things is to say my rack has to work, but I have some spare racks as well.

[00:42:35.05] Reiner Pope

You can math that out with the tax of reliability here is only like 10%, that’s pretty good. But that relies on someone coming and servicing the part within a day or something like that. If you say they’re going to service it never, then I think you actually can get where you want to be, but maybe with 100% tax on reliability rather than 10%. For example, if you think the average lifetime of a chip is in the range of three to five years. That means if I deploy twice as many chips, then three to five years from now, half of them will still work.

[00:43:06.02] John Collison

Also, the burn-in is particularly failure-y. How about the cooling?

[00:43:11.18] Reiner Pope

Most of the challenge... I guess this is actually really a data center design aspect. At the rack level, the challenge of cooling is just getting the heat out as quickly as possible out of the rack into the cooling network. How you get it out of the spaceship, other people would know that better than I do.

[00:43:30.22] John Collison

That seems to be the main objection, but I don’t know.

[00:43:36.19] Reiner Pope

I think it’s if you think the cost of repair is that you need to have deployed twice as many chips, then it’s a trade-off of the capital of the chips versus the power saving.

[00:43:46.14] John Collison

Exactly, the repair thing, it feels like, can be solved, because also, I think part of the bet, part of Elon’s claim, is that we will just be so power-limited that you have no option but to go to space. People can argue about that, but were that to be the case, then yes, it’s like you can get power in space, and you cannot on Earth, so you might as well go there. Whereas like the cooling is a more fundamental, “Does the product actually work at all?”

[00:44:18.03] John Collison

Reiner thinks about AI the unglamorous way: compute, systems architecture, and what it takes to run models reliably at scale. If you’re building an AI product, the business model similarly has a ton of unglamorous complexity. You’re not just selling AI, you’re monetizing consumption across API calls, tokens processed, GPU hours. Stripe Billing is a scalable system for usage-based billing. It lets you launch token-based pricing, subscriptions, credits, hybrid models, whatever you want. You can create revenue models based on usage without rebuilding your pricing system every six months. If you’re building an AI product, Stripe Billing is worth a look.

[00:44:55.04] John Collison

What are your AI predictions for 2026?

[00:44:57.18] Reiner Pope

What I’m really excited about is just being able to… I’m still excited about the coding. This is what we do as a company. It’s what many others do as a company as well. The one aspect of this is expanding into more domains. For example, in where we spend our time. We as a company, we write Rust, we write Verilog, we write Python.

[00:45:26.02] John Collison

No Haskell?

[00:45:26.18] Reiner Pope

No, there’s a story there. I used to love Haskell. Rust is my current favorite. Mutation is good. The models are extremely good at Rust and Python. They’ve done a lot of RL on them. They have not done as much RL on Verilog. They’ve done almost none on, “Write me a markdown file that describes a chip architecture.” How do you even RL on that? You have to say, what is a good chip architecture? You have to somehow say whether that’s a good result or not. I think one of the things the labs are doing is trying to broaden what they’ve done RL on, source it from customers and so on in order to make it less spiky, fill out the gaps between the spikes.

[00:46:17.04] John Collison

I presume the labs would love to work with you on improving the models by doing RL on this specific task. However, it’s also special.

[00:46:29.14] Reiner Pope

How does that make sense for us?

[00:46:30.15] John Collison

You’re a special sauce. Do you want to come up with some AI approaches but keep them proprietary?

[00:46:39.05] Reiner Pope

We’ve looked at a few different aspects here. There’s the… What we’re able to do by ourselves, our business is not training models. We do it in order to do the research on numerics, but actual production models we don’t do. The biggest mileage is on the RL, and it’s not something we can really do ourselves. We’d love it if we could have a custom model just for us, but that doesn’t seem to be—

[00:47:05.02] John Collison

Well you could, right?

[00:47:05.06] Reiner Pope

The terms we’ve been offered by labs so far have not been on those terms, but—

[00:47:08.16] John Collison

Because you have to share the IP back.

[00:47:10.06] Reiner Pope

The way they prefer to do it is that they put it into their mainstream model because it’s good for them.

[00:47:16.23] John Collison

Which, obviously, you don’t want to do. How do you think… What does you using AI to design a model do you think look like? Because this is actually an interesting sight glass into a weak version of recursive self-improvement where we’re using the AIs to develop better AIs. I’m curious, what you think that looks like? Is it your own proprietary recursive models? What else? Is there day-to-day AI usage that’s load-bearing?

[00:47:44.20] Reiner Pope

The stuff that is available today and I think will become even better, very quickly is just the stuff that looks most like software. Writing Verilog, running tests, running continuous integration, and so on. That is a big fraction of the development time in a chip. It’s probably 9, 12, 15 months. There’s some stuff that’s downstream of that, which is physical design, which is, you take that Verilog and you generate the gates and the polygons. We don’t have a clear path for… It’s not at least the most obvious thing is not clear for how to compress that.

[00:48:24.21] Reiner Pope

The goal, can you tape-out a chip in one month? One month would be the goal. In theory, you could compress all of the logic design and design verification down to a short amount of time just by continuing on the same path we’re doing now. But if you wanted to take the physical design down, that has to leave code, you’re now doing like graphical interfaces and saying I want to play stuff and so on. Actually, there has been work on this even prior to LLMs, which is a specific model trained for that particular problem.

[00:48:56.01] Reiner Pope

I think the vendors, which is like Synopsys and Cadence, probably should move in that direction. Most of the focus has not been, do it faster; it’s been, do it with higher quality. But that is a big bottleneck on, can I have a new chip every month? There’s just the practical thing of a new chip every month doesn’t really make sense because then if I’m deploying... If it takes me a year to populate a data center, that means I’m going to have different chips in different corners of the data center.

[00:49:23.16] John Collison

Yes. When you talk about one month to tapeout, so you do all this work to ultimately produce a file. Everything TSMC then does, it’s not entirely in software. Is there some type-setting that has to happen of moving stuff around? But what happens when you send your files to TSMC? Then what?

[00:49:45.19] Reiner Pope

They create a mask. That is where the ASML tools come in. A mask is really just a stencil. You shoot the lasers through the mask or the x-rays through the mask and then that produces the different P-type and N-type semiconductors. They produced the mask. That is the expensive part. They’re building up these 15 or so metal layers. They placed on the silicon and then there are different layers of metals which connect all the transistors together. They do that on a wafer. It happens on a stepping basis. There’s a maximum size of chip you can build which is constrained by this machinery.

[00:50:30.18] John Collison

The wafer stepper is part of the ASML special sauce, right?

[00:50:34.16] Reiner Pope

There are probably some important alignment requirement there.

[00:50:37.13] John Collison

I think I remember that being quite like the classic manufacturing throughput problem. I think they’ve done a lot of work on optimizing that.

[00:50:45.09] Reiner Pope

They tape that, so then you just produce hundreds of copies of your chip. You have to test it because there are defects. You typically, I think the average rates really depends on process and so on, but small single digit number of defects per chip. You test the chip and see whether it has any defects in it. Many chips are designed to be able to tolerate a few defects, and so you need to configure it to tolerate the defects. Now you have a die that by itself works. Then you need to package it. You put it in a package together with memories, typically that’s the HBM, and then maybe you escape the wires to connect to other chips.

[00:51:25.22] John Collison

How long does it take to make a mask?

[00:51:29.07] Reiner Pope

What we see is time from tape-out to first chips back. Again, depends on node, but it’s ballpark four or five months.

[00:51:37.01] John Collison

So tape-out is just sending the file?

[00:51:40.19] Reiner Pope

We consider tape-out as sending the file, and then there’s a whole process of you make the masks for all the layers, and then, actually just producing the chips.

[00:51:48.14] John Collison

Producing the masks and producing the chips happens after tape-out.

[00:51:51.12] Reiner Pope

That’s right.

[00:51:51.23] John Collison

I see. Is the term tape-out from like you send a magnetic tape with the instructions or something?

[00:51:57.22] Reiner Pope

It could be. I was in software when the software was created.

[00:52:02.01] John Collison

I was curious about the tape actually means. It feels like… When I think about AI predictions, one thing I’m really struck by is how, still in 2026, every time you open a chat window, it’s contextless. It’s got no memory. Now, to be fair, it’s like, guys, it’s been four years. Not even four years. It’s been three and a half years. Just calm down, we’ll get there. But I also interpret a lot of the current enthusiasm for OpenClaw and all that stuff as this super hacky backdoor into state management where your little claw will write a markdown file of what it’s doing and then look at that markdown file the next time and things like that. But it just feels like state management and memory is going to be a huge deal and that will really change the character of AI products.

[00:52:57.20] Reiner Pope

It’s really interesting. Long context is one of the biggest bottlenecks on speed of the model. Every single token you generate, it reads through all of the previous tokens, or maybe it reads through a subset of them, but reads through a lot of the previous tokens you’ve written. Memory bandwidth for that is really constraining. You can think of model-level ways to solve that problem, which is to say maybe I can compress it into fewer bytes or something like that. But it’s interesting that the most effective way to solve it has been—it’s really a combination of everything—but the most effective way to solve it has been once you hit your 300,000 token limit, have the model go back through it and compact.

[00:53:44.12] John Collison

This is kind of what OpenClaw is doing. It’s compacting everything you’ve done. But it’s funny that it’s so manual.

[00:53:52.11] Reiner Pope

I think—

[00:53:54.11] John Collison

Manual is the wrong word. It’s so primitive.

[00:53:57.02] Reiner Pope

It’s maybe because it’s so controllable. If you want to iterate on how you compact, you give a different prompt, and you say, “Compact this way, compact that way.” You can iterate that on that in seconds or minutes. Whereas if you’re trying to do some iteration on the model level, where you say, “Now I’ve got a different model architecture,” it’s going to take you months to train and launch something.

[00:54:16.09] John Collison

Yes. Any other AI predictions?

[00:54:18.16] Reiner Pope

I’m generally just interested in what makes models cheaper and faster. Just at the model architecture level, really tied into this context thing. I think the context size will stay ballpark the same where it is, maybe a few times larger, but the parameter count will go up. Parameter count should grow much faster than context length actually, just because of the underlying physics of what’s available.

[00:54:41.23] John Collison

Though, has that been the story? Would that be a reacceleration of parameter count? Because it feels like we’ve leveled off slightly in the last year or two, and instead we’ve been focusing on more and better RL.

[00:54:55.21] Reiner Pope

Parameter count or thinking tokens, I guess. Those are available, but the context length is struggling to grow.

[00:55:05.23] John Collison

But you think we… When you say context length is struggling to grow, but you’re saying we keep context the same length.

[00:55:12.01] Reiner Pope

Keep context the same length.

[00:55:13.05] John Collison

But we’re better at working with large context. Is that what you’re saying?

[00:55:16.16] Reiner Pope

Just have application-level interventions to manage large context, like compacting.

[00:55:22.10] John Collison

Because I think everyone’s had the experience currently of the chat conversation and the further down in the chat you get.

[00:55:28.21] Reiner Pope

It just gets looser and—

[00:55:30.01] John Collison

Yeah, sloppy. It’s just like really sloppy by the end, and it’s like making mistakes, whatever. You’re saying we start to do better with large contacts. I buy that. When will I be typing into a chat window, and it is a MatX chip underneath it, powering it?

[00:55:45.09] Reiner Pope

Tape-out in under a year. That means chips available end of year. Ballpark.

[00:55:51.17] John Collison

That’s exciting. So in 2027, I will be seeing very high performing chats as a result of your chips.

[00:55:57.13] Reiner Pope

In the 1% experiment of the users or something like that.

[00:55:59.23] John Collison

Exactly. I need to find a way to finagle myself into the A/B test. MatX is 100 people?

[00:56:07.22] Reiner Pope

That’s right.

[00:56:09.11] John Collison

How have you gone about building the team, the culture?

[00:56:14.08] Reiner Pope

What we have on the team is hardware, mostly hardware, but a big software team and also a big ML team. I think the ML team is quite unusual in what we ask them to do. When you look at a typical ML team in an AI chip company, it will be what I might say ML engineering or ML performance. They’re writing kernels that actually just use your hardware as well on a given model. There’s a missed opportunity there. If you’re saying all we do is we take other people’s models, and we write kernels for them, you’re optimizing this, but you can’t optimize this at the same time. We want to optimize the whole thing at the same time, real code design.

[00:56:57.09] Reiner Pope

Our ML team is actual real ML research. What they do every day is they train small LLMs from scratch focusing on numerics and attention. This has really, really helped us make an interesting product. It’s shown up most strongly in our numerics. Often what you see when people design numerics is they say, back when Float32 was popular, it would be, “I’m going to follow the IEEE standard.” Now it is like, follow the Open Compute standard.

[00:57:35.19] Reiner Pope

There are lots of little details where you say things like maybe, “What’s the rounding mode I’m going to use?” Like, “Round to the nearest even,” or something like that, which is like the best known standard for how to round. We want to cut corners anywhere we can. Maybe don’t do the best rounding, maybe don’t do the… Don’t get all the corner cases correctly. That’s a very scary proposition if you’re just making those choices blind, but if you have the benefit of a research team who can back you up as you do that, it’s really powerful, and it’s really interesting that we can make some sloppy choices in these cases.

[00:58:11.21] John Collison

I feel like often technical advances come through better iteration loops. A favorite example of this I found recently was that the Wright brothers actually had a failed season before first flight. First flight was in 1904, and they were down in Kitty Hawk in 1903 and not making that much progress. They went back to Ohio, and they had a wind tunnel, and they were testing their design in the wind tunnel. You can imagine not a lot of wind tunnels in 1904. They did a lot of wind tunnel testing and their successful flight was after that. Is this something you’re focused on where, to get better chips, you allow for a better testing and iteration loop, and what does that look like?

[00:58:55.13] Reiner Pope

I think this mostly happens in the architecture and product definition stage. Maybe even more generally, I think AI chips seem to live or die by product definition and architecture. What is the most extreme form of fast iteration is doing it in your head. Can you map a model to hardware in your head? Can you estimate the performance of what it is in your head? You’re not going to be 100% perfect, but maybe you can prove some lower bound in performance.

[00:59:24.16] Reiner Pope

The simplest possible thing is my model has a trillion parameters, my device can do a billion multipliers per second, so it takes 1,000 seconds to run or something like that, just to do that simple division. But then there are much more complicated things like we tend to look at resource balances, like how many memory fetches do I need to do per a multiplier or something like that. At least the way I like to do design and architecture and optimization is to be able to estimate the performance to within about 30-40% before even typing anything in at all. We’ve tried to do that a lot. A lot of our architecture comes from there.

[01:00:09.16] Reiner Pope

Then the next stage of iteration is… That’s on the performance side. This also happens on the circuit design side as well. Can you take a circuit and say what is the gate count on that? A 16-bit multiplier has approximately 16 squared mini-gates, and you can do that for more complicated things by sorting networks and so on. We already have a pretty good idea of the costs and speeds of things at that point after doing these calculations. Then what we tend to do as the next step of iteration is on the ML side, we run model experiments. You get iteration speed just by having small models mostly. On the hardware side, we use performance simulators to do the next level of detail to make sure we’re seeing all the things we want to see.

[01:00:56.19] John Collison

This idea that the best iteration is in your head is reminding me of Jeff Dean’s numbers... Do you have your equivalent of that numbers every MatX?

[01:01:06.15] Reiner Pope

We have go/gates in our company, which says, what is the cost of an XOR gate, an AND gate, a full adder, SRAM bitcell, and so on.

[01:01:15.18] John Collison

You want people to be working with that stuff in their head and have an intuitive sense for it because it leads to better iteration. What is the pitch to someone joining MatX?

[01:01:25.23] Reiner Pope

I think if you are someone who likes optimizing, just optimize something—software, hardware, factorial, whatever—if you’re trying to fit something into the smallest budget possible, I think it’s a pretty exciting place to be. I think hardware companies in general are really exciting because you have such a broad range of skills of people on the team. You have software people, you have hardware people, you’ve got physical design, you’ve got people who are just looking at the insertion force of a card into a rack. So there’s so much discussion and learning you can do.

[01:02:05.23] Reiner Pope

I think MatX in particular, we really care about this and I think we extended it all the way up into the application and the machine learning as well. Really, really interesting technical problems, and I think just generally there’s lots of interesting people to talk to.

[01:02:26.22] John Collison

Yes. Presumably in terms of impact, if you can design a meaningfully higher throughput chip, a 20% higher throughput chip means 20% more AI is happening. If the bottleneck is elsewhere, like power or something like that or cost, you actually just are meaningfully increasing the amount of intelligence in the world, which is presumably exciting to people.

[01:02:48.04] Reiner Pope

I think this shows up both as just can it apply in more applications, as well as just how smart is the model.

[01:02:56.23] John Collison

Yes. How about Rust?

[01:02:59.06] Reiner Pope

A previous project I worked on at Google, we did a lot of Haskell. I did Haskell when I was at school. I loved it. Very principled, very interesting. I like Haskell, but I also like making stuff fast. The question is, what is the first thing you want to do? You want to be able to modify your memory. Haskell, you jump through hoops to do that. Maybe I just want a language that is functional programming that lets me modify my memory. I think Rust has a lot of the nice things which are like type classes or traits and a rich type system.

[01:03:35.12] Reiner Pope

One of the things that we have done, interesting ways we use it at MatX are the range of data types that you express on software. What are the integer types? Int32, int64, int8, maybe that’s all you care about. But it turns out in hardware, you care about every single bit, and so you want to use 17, 18, 19-bit integers. That is quite natural to express, and we build up a whole ecosystem of rich hardware data types in Rust as well.

[01:04:10.23] John Collison

Has Rust beaten Go for the position of performant type programming language with modern features or do they actually address different?

[01:04:20.04] Reiner Pope

There’s what the Rust marketing will say, which is “safe without a garbage collection,” which I think is a real… It is the objective thing that you can say is different, but sort of buries the lead, which is also just it’s got nice type system features that Go doesn’t have. Why is garbage collection… Why does it matter at all? People often focus on the time it takes to run a garbage collector, but the other thing is that every time you allocate an object, you’ve got the object, and then you’ve got the garbage collector header at the beginning. It uses a lot more memory as well. If you want to design some, I don’t know, data structure that uses the right amount of memory rather than a bit more than…

[01:05:04.19] John Collison

I hadn’t realized that in Rust you’re allocating your memory manually versus in Go you have—

[01:05:09.07] Reiner Pope

That’s right.

[01:05:10.19] John Collison

I didn’t realize that. And you prefer that for what you’re doing.

[01:05:13.05] Reiner Pope

I just really like dealing with the details. Like, you give me a puzzle, and I’ll be like, “Let me solve every single piece of it.” That tickles that part of my mind with Rust.

[01:05:21.13] John Collison

It seems like you’re a fan of optimization generally. Is that a fair characterization?

[01:05:25.17] Reiner Pope

Yeah.

[01:05:26.03] John Collison

Where else have you… Chip optimization is one domain, where else?

[01:05:31.00] Reiner Pope

When one of the really exciting things I found about working at Google is that the whole Google code base is available, and you can look at how does a memory allocator work, how does a mutex work, how does a HashMap work, any of those things, and you can go and look inside the implementations. Google has excellent implementations of those, some of the best you could write.

[01:05:59.07] Reiner Pope

One of the things I did on my nights and weekends when I was at Google was just to go find those implementations, write a benchmark. How many nanoseconds does it take to allocate eight bytes of memory? Can I make that faster? Can I maybe I inline this function? Maybe I look at the assembly and say, “It looks like there’s a few memory moves here or there are some registers that are being used that I don’t need in the fast path and in the slow path. Can I do something there?” I don’t know, that was always my just fun and learning activity.

[01:06:32.05] Reiner Pope

Being outside of Google, I feel, I probably could have done this inside of Google as well, but outside of Google, I felt the luxury to be able to talk about these results as well. One of the things I’ve looked at recently is just hash tables are used so much. One prompt for me was, if I wanted to design custom CPU instructions for accelerating hash tables… Hash tables are one of the most common things. I’m looking at them up and writing them all the time. What would the optimal CPU be for that? Following down that chain is like, what is the best hash table implementation in the first place?

[01:07:14.08] Reiner Pope

I spent some time looking at different SIMD implementations. There’s this really cool technique called ‘cuckoo hashing’ where you hash into two different locations, and then you use the bucket which is less full. It’s been in the literature for decades and yet the best hash table implementations don’t use it because it’s somehow not practical.

[01:07:39.03] John Collison

Why is it not practical?

[01:07:40.13] Reiner Pope

Practical hash tables are these days considered to be ones that use SIMD vector instructions to scan eight buckets at a time. The way cuckoo hashing is normally described is I look up one bucket here and one bucket there. I’m not using the vector instructions. Vector instructions are much faster than scalar instructions and so there’s a missed opportunity. Again, just take the two good ideas and stick them together, do vector instructions on cuckoo hashing. You have to be careful to get the details right, but if you get it right, you can actually just wing it.

[01:08:16.09] John Collison

Sorry, is your claim that one could design a custom CPU that has way better hash table performance, or even on current chips, you could get way better hash table performance?

[01:08:28.06] Reiner Pope

Both. I’m interested in what you can design in custom hardware, but MatX doesn’t make CPUs. We’re not going to make CPUs.

[01:08:36.23] John Collison

You could. New line of business.

[01:08:38.04] Reiner Pope

We just want to focus on shipping one product well for the time being.

[01:08:43.20] John Collison

Fair. Good answer.

[01:08:44.21] Reiner Pope

I think it’s an interesting exercise, but I don’t get to feel the endorphins of seeing the number go down. I first did this on just Intel CPUs. You can get better performance than some of the best hash table implementations available using cuckoo hashing on Intel CPUs.

[01:09:04.17] John Collison

What are examples of workloads that are really hash table read intensive? I know everything,

[01:09:12.03] Reiner Pope

JavaScript, I guess, but… It’s a tricky exercise because when you really think about it, you’re like, “Did I really need a hash table there? I probably didn’t, but I just reach for it all the time.”

[01:09:25.22] John Collison

But you can go to the Google JavaScript team and probably help them eke out better performance in the Chrome JavaScript engine?

[01:09:31.19] Reiner Pope

Potentially. I’m not going to spend my time on that.

[01:09:35.01] John Collison

If they’re listening to this podcast, just a free idea from Reiner. Explain the dragon.

[01:09:41.01] Reiner Pope

This is from a book that, when I was working on the JAX team… The JAX team is one of the ML infrastructure teams at Google. I was there as the most recent team before I left.

[01:09:50.17] John Collison

I’m sorry, what does the JAX team do?

[01:09:52.08] Reiner Pope

The JAX team develops… This is Google’s new, more modern version of TensorFlow or a competitor to PyTorch. It’s how you write models in Python to run on TPUs. A big part of the JAX team, however, is to say, we have JAX, the technical artifact. Can we help enable users to actually use it really well and get high performance? Ultimately that became, who are the users? It’s people writing LLMs. How do you get good performance on LLMs?

[01:10:20.22] Reiner Pope

Really, really strong team, the JAX team at Google, although as with a lot of brain people are now elsewhere as well. We developed a lot of the different techniques for how to lay out models efficiently on many chips. Ultimately, some people at Google—and I contributed after I left Google—wrote this guide called How To Scale Your Model, how to run an LLM as fast as possible. It is the main reference for how to get high performance on TPUs. There is now also a GPU version of this as well. It’s a dragon because it’s How to Train Your Dragon.

[01:10:52.22] John Collison

I see. Last question. People might not have thought that there’s room for new chip companies. It might have seemed unusual or very hard. You guys, it seems like a very good approach with that. Where do you think are other opportunities for companies to be started here in 2026? Where do you think people should be looking for entrepreneurial opportunities or just technical challenges that haven’t been properly addressed?

[01:11:21.13] Reiner Pope

More labs, I think, is still interesting. Just can we do more on model architecture is always interesting.

[01:11:27.15] John Collison

You think we have not fully explored model architecture space?

[01:11:30.05] Reiner Pope

The Frontier Labs have done a pretty good job of exploring it, but I think, as the hardware changes, the shape of the model should change for sure.

[01:11:40.14] John Collison

Presumably you’re not thinking like, yet another Frontier Lab pursuing the same architecture. You think there’s probably off the wall looking architectures that will actually make a lot of—

[01:11:50.21] Reiner Pope

I think there’s a little bit off the wall for sure.

[01:11:53.19] John Collison

Do you have a specific architecture in mind?

[01:11:55.19] Reiner Pope

My mentality is always sticking within the transformer family, but what are the constraints that are currently available, currently imposed that you could lift? For example, one of the things is there’s this idea, when you’re doing transformer inference, you do pre-fill that is processing what the user said to you, and then there’s decode which is generating the response to that. Those are totally different in pretty much every aspect of how they actually run. One runs a step at a time, the other one runs really in parallel.

[01:12:26.03] Reiner Pope

There is this somewhat artificial constraint today that those are the same model that’s doing both. Maybe lift that constraint. Another example would be, there’s this idea that the model that you… This is more fundamental constraint that you have to train the same model as you serve. But again, training is very different from serving. At training, it’s very compute intensive. At serving, it’s more memory bandwidth intensive. Maybe, is there a way you can make a model that when you use it at inference time, it increases the amount of computing it does to use some of the available resources?

[01:12:59.11] John Collison

Makes sense. Reiner, thank you.

[01:13:01.20] Reiner Pope

Pleasure.

Thanks for reading! Subscribe for free to receive new posts and support my work.

Subscribe

3

1

Share
