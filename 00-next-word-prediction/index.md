# Chapter 00: The Next-Word Guessing Game

<nav aria-label="Table of Contents">
  <p>
    <strong>Table of Contents:</strong> 
    <a href="#step-1">1. Intuition</a> &bull; 
    <a href="#step-2">2. Bridging Question</a> &bull; 
    <a href="#step-3">3. Exact Math</a> &bull; 
    <a href="#step-4">4. Origin</a> &bull; 
    <a href="#step-5">5. Toy Example</a> &bull; 
    <a href="#step-6">6. Core Takeaway</a>
  </p>
</nav>

<hr>

## Step 1: 3-Year-Old Intuition (The Animal Mystery Box)

> [!INTUITION] The Animal Mystery Box
> Imagine you and your best friend are playing a game with a mystery box.
> 
> You give your friend a clue: <kbd>"It is fluffy..."</kbd>  
> Your friend thinks: *"Could it be a bunny? A kitten? A cloud?"*
> 
> Then you whisper a second clue into their ear: <kbd>"It is fluffy... and it says meow!"</kbd>  
> 
> Instantly, your friend laughs and shouts: <samp>"Kitten!"</samp>
> 
> Did your friend know the answer before the word "meow"? No! But every single clue narrowed down the possibilities until only one answer made sense.
> 
> A Large Language Model (LLM) is playing this exact same game every single millisecond. You give it prompt tokens: <kbd>"The"</kbd> <kbd>"sky"</kbd> <kbd>"is"</kbd> <kbd>"very"</kbd> and it looks through its giant toy chest of words to pick the next one: <samp>"blue"</samp>.

---

<h2 id="step-2">Step 2: The Bridging Question (From Whispers to Numbers)</h2>

How does a computer play this guessing game mathematically?

A computer cannot "feel" what word comes next. It cannot say: *"Well, kitten feels cozy."* 

Instead, a computer needs a **mathematical ruler**. It must assign a precise numerical score—a **probability** between 0% and 100%—to every single word in human language, measuring how likely that word is to follow the previous clues.

---

<h2 id="step-3">Step 3: The Exact Math &amp; Formulas</h2>

### 1. The Vocabulary Box ($V$)

Before the model can make a guess, we must write down a master catalog of every word or subword it is allowed to speak. We call this catalog the **Vocabulary Set**, denoted by the symbol $V$:

$$V = \{\text{"the"}, \text{"cat"}, \text{"sat"}, \text{"on"}, \text{"mat"}, \dots\}$$

The total count of distinct words in this box is called the **Vocabulary Size**, written as $|V|$:
- In toy models: $|V| = 4$ or $|V| = 10$.
- In production models (GPT-4, LLaMA-3): $|V|$ is typically between $32{,}000$ and $128{,}000$ tokens.

---

### 2. The Conditional Probability Rule

When the model has already observed a sequence of $t-1$ words:

$$w_1, w_2, \dots, w_{t-1} \quad (\text{often abbreviated as } w_{<t})$$

it calculates the probability that the next word at position $t$ is a specific candidate word $w \in V$. We write this as:

$$P(w_t = w \mid w_1, w_2, \dots, w_{t-1})$$

The vertical bar $\mid$ translates into plain English as **"given that we already observed"**.

Every valid probability distribution in an LLM must satisfy two unbreakable mathematical laws:

1. **Non-negativity & Boundedness**: No word can have a negative chance, and no word can have more than a 100% chance:
   $$0 \le P(w_t = w \mid w_{<t}) \le 1 \quad \text{for all } w \in V$$

2. **Total Probability Conservation**: If you add up the probabilities of every word in the entire dictionary, the sum must equal exactly $1$ (100% of the pie):
   $$\sum_{w \in V} P(w_t = w \mid w_{<t}) = 1$$

---

### 3. The Chain Rule of Probability

How does an LLM generate an entire essay, a poem, or a computer program? It strings individual next-word guesses together using the **Chain Rule of Joint Probability**:

$$P(w_1, w_2, \dots, w_T) = P(w_1) \times P(w_2 \mid w_1) \times P(w_3 \mid w_1, w_2) \times \dots \times P(w_T \mid w_1, \dots, w_{T-1})$$

Using compact mathematical product notation ($\prod$):

$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$$

> [!MATH] Decoding the Mathematical Symbols
> - $\sum$ (Capital Sigma): Means **"Add them all up"**.
> - $\prod$ (Capital Pi): Means **"Multiply them all together"**.
> - The total probability of a paragraph is the chance of the first word, multiplied by the chance of the second word given the first, multiplied by the third word given the first two, and so on until the final period.

### Why Do We Calculate the Joint Probability of a Full Sentence?

Why can't an LLM simply stop at guessing the next single word? Why do researchers care about the joint probability $P(w_1, w_2, \dots, w_T)$ of the entire paragraph?

There are three foundational reasons:

1. **Training the Brain (Maximum Likelihood Estimation)**:  
   When an LLM trains on billions of sentences from libraries and the internet, how does it know whether its parameters are improving? It measures the joint probability it assigns to real human sentences. Training an LLM is mathematically defined as adjusting internal weights so that $P(w_1, \dots, w_T)$ is as close to $1.0$ as possible for coherent language, while assigning near-zero probability to scrambled words. (This forms the exact foundation of Cross-Entropy Loss in Chapter 14).

2. **Comparing and Ranking Competing Thoughts**:  
   When generating text or translating between languages, the model often considers multiple candidate sentences:
   - Candidate A: <samp>"The cat sat on the warm rug."</samp>
   - Candidate B: <samp>"The cat sat on the warm rug rug."</samp>  
   Evaluating the joint probability across all tokens allows the model to score and rank complete hypotheses, picking the one that makes the most cohesive sense as a whole.

3. **Preventing Short-Sighted "Greedy" Traps**:  
   A word that looks highly probable in the short term might lead into a grammatical dead-end two words later. Calculating or approximating the joint probability over sequence paths allows algorithms like Beam Search to steer clear of dead ends.

---

### Mathematical Symbol Breakdown Table

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 0.1:</strong> Formal mathematical symbols, dimensions, and operational definitions.</caption>
  <thead>
    <tr>
      <th align="left">Symbol</th>
      <th align="left">Formal Name</th>
      <th align="left">3-Year-Old Meaning</th>
      <th align="left">Concrete Toy Example</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>$V$</td>
      <td>Vocabulary Set</td>
      <td>The big toy box of words the model knows</td>
      <td>$V = \{\text{"I"}, \text{"love"}, \text{"ice"}, \text{"cream"}\}$</td>
    </tr>
    <tr>
      <td>$|V|$</td>
      <td>Vocabulary Size</td>
      <td>How many word blocks are in the toy box</td>
      <td>$|V| = 4$ words</td>
    </tr>
    <tr>
      <td>$t$</td>
      <td>Time Step / Position</td>
      <td>The index of the word we are guessing right now</td>
      <td>$t = 4$ (the fourth word)</td>
    </tr>
    <tr>
      <td>$w_t$</td>
      <td>Token at Step $t$</td>
      <td>The candidate word being considered</td>
      <td>$w_4 = \text{"cream"}$</td>
    </tr>
    <tr>
      <td>$w_{<t}$</td>
      <td>Prefix Context</td>
      <td>All previous clues whispered into the funnel</td>
      <td>$w_{<4} = (\text{"I"}, \text{"love"}, \text{"ice"})$</td>
    </tr>
    <tr>
      <td>$P(w_t \mid w_{<t})$</td>
      <td>Conditional Probability</td>
      <td>The model's belief meter ($0.0$ to $1.0$) for that word</td>
      <td>$P(\text{"cream"} \mid \text{"I love ice"}) = 0.90$</td>
    </tr>
    <tr>
      <td>$\sum$</td>
      <td>Summation (Sigma)</td>
      <td>Add up all slices of the probability pizza</td>
      <td>Sum of all 4 word probabilities $= 1.00$ (100%)</td>
    </tr>
    <tr>
      <td>$\prod$</td>
      <td>Product (Capital Pi)</td>
      <td>Multiply the chances of each step along the chain</td>
      <td>$0.50 \times 0.40 \times 0.30 \times 0.90 = 0.054$</td>
    </tr>
  </tbody>
</table>

<br>

<details>
<summary><strong>Symbol Glossary (Definition List)</strong></summary>

<dl>
  <dt><strong>Vocabulary Set ($V$)</strong></dt>
  <dd>The finite set of all unique words, subwords, or characters recognized by the tokenizer.</dd>
  
  <dt><strong>Vocabulary Cardinality ($|V|$)</strong></dt>
  <dd>The total count of tokens in $V$. Modern LLMs typically range from 32,000 to 128,000 tokens.</dd>
  
  <dt><strong>Sequential Index ($t$)</strong></dt>
  <dd>The discrete time step or token position along the sequence axis ($t \in \{1, 2, \dots, T\}$).</dd>
  
  <dt><strong>Target Token ($w_t$)</strong></dt>
  <dd>The specific word chosen or predicted at sequence position $t$.</dd>
  
  <dt><strong>Prefix History / Context ($w_{<t}$)</strong></dt>
  <dd>The ordered tuple of all tokens preceding step $t$: $(w_1, w_2, \dots, w_{t-1})$.</dd>
  
  <dt><strong>Conditional Probability ($P(w_t \mid w_{<t})$)</strong></dt>
  <dd>A real scalar between $0.0$ and $1.0$ satisfying $\sum_{w \in V} P(w \mid w_{<t}) = 1.0$.</dd>
</dl>
</details>

---

<h2 id="step-4">Step 4: Where Does This Formula Come From?</h2>

Why do all modern LLMs formulate language generation as a chain of conditional probabilities rather than predicting the whole sentence at once?

<details>
<summary><strong>Historical Origins: Andrey Markov (1913) & Claude Shannon (1948)</strong></summary>

<p>In 1913, Russian mathematician <strong>Andrey Markov</strong> spent months working by candlelight with a copy of Alexander Pushkin's verse novel <em>Eugene Onegin</em>. He meticulously counted 20,000 Russian letters by hand to prove that the probability of a letter being a vowel depended strongly on whether the previous letter was a consonant.</p>

<p>This proved that language has <strong>temporal dependence</strong>: letters and words do not appear randomly; each token depends on what came immediately before it. This mathematical structure became known as a <strong>Markov Chain</strong>.</p>

<p>Thirty-five years later, in 1948, <strong>Claude Shannon</strong> published his landmark paper <em>"A Mathematical Theory of Communication"</em>. Shannon formalized human language as a statistical stochastic process where each successive symbol is chosen according to conditional probability distributions conditioned on preceding symbols.</p>
</details>

<br>

<details>
<summary><strong>What Broke When Researchers Tried Simpler Ideas?</strong></summary>

<p>Why can't we just assume words are independent? That is, why not compute:</p>
$$P(w_1, w_2, w_3) \stackrel{?}{=} P(w_1) \times P(w_2) \times P(w_3)$$

<p>If words were independent, the sentence <em>"The dog bit the man"</em> would have the exact same probability as <em>"Bit man dog the the"</em>, because both contain identical words! Language relies entirely on word order and context. Therefore, conditioning on the past prefix $w_{<t}$ is mathematically essential.</p>
</details>

---

<h2 id="step-5">Step 5: Concrete Toy Example (Step-by-Step Arithmetic)</h2>

Let's walk through the exact arithmetic using our tiny toy language:

$$V = \{\text{"I"}, \text{"love"}, \text{"ice"}, \text{"cream"}\} \quad (|V| = 4)$$

We want to calculate the probability of generating the complete sentence:  
**"I love ice cream"**

<figure>
<pre>
Step 1: P("I")                                        = 0.50  (50% chance)
             │
Step 2: P("love"  │ "I")                              = 0.40  (40% chance)
             │
Step 3: P("ice"   │ "I", "love")                      = 0.30  (30% chance)
             │
Step 4: P("cream" │ "I", "love", "ice")               = 0.90  (90% chance)
──────────────────────────────────────────────────────────────────────────
Full Joint Probability: 0.50 × 0.40 × 0.30 × 0.90     = 0.054 (5.4%)
</pre>
<figcaption><strong>Figure 0.1:</strong> Sequential probability conditioning across four time steps.</figcaption>
</figure>

---

### Step-by-Step Probability Distribution at Step 4

At step $t=4$, after hearing *"I love ice"*, the model evaluates every candidate word in $V$:

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 0.2:</strong> Candidate word probability distribution conditioned on prefix context <kbd>"I love ice"</kbd>.</caption>
  <thead>
    <tr>
      <th align="left">Candidate Word $w \in V$</th>
      <th align="right">Conditional Probability $P(w \mid \text{"I love ice"})$</th>
      <th align="right">Percentage</th>
      <th align="center">Visual Probability Bar</th>
      <th align="left">Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><strong><kbd>"cream"</kbd></strong></td>
      <td align="right"><strong>0.90</strong></td>
      <td align="right"><mark><strong>90%</strong></mark></td>
      <td align="center"><meter min="0" max="1" value="0.90" optimum="0.8">90%</meter></td>
      <td align="left">Clear favorite</td>
    </tr>
    <tr>
      <td align="left"><strong><kbd>"ice"</kbd></strong></td>
      <td align="right">0.05</td>
      <td align="right">5%</td>
      <td align="center"><meter min="0" max="1" value="0.05">5%</meter></td>
      <td align="left">Unlikely ("ice ice")</td>
    </tr>
    <tr>
      <td align="left"><strong><kbd>"love"</kbd></strong></td>
      <td align="right">0.03</td>
      <td align="right">3%</td>
      <td align="center"><meter min="0" max="1" value="0.03">3%</meter></td>
      <td align="left">Unlikely ("ice love")</td>
    </tr>
    <tr>
      <td align="left"><strong><kbd>"I"</kbd></strong></td>
      <td align="right">0.02</td>
      <td align="right">2%</td>
      <td align="center"><meter min="0" max="1" value="0.02">2%</meter></td>
      <td align="left">Very unlikely ("ice I")</td>
    </tr>
    <tr>
      <td align="left"><strong>Total Sum ($\sum$)</strong></td>
      <td align="right"><strong>1.00</strong></td>
      <td align="right"><mark><strong>100%</strong></mark></td>
      <td align="center"><progress max="100" value="100">100%</progress></td>
      <td align="left"><strong>Conserved Probability</strong></td>
    </tr>
  </tbody>
</table>

---

### Step-by-Step Multiplication

Now we multiply the 4 probabilities together using the Chain Rule:

$$P(\text{"I love ice cream"}) = 0.50 \times 0.40 \times 0.30 \times 0.90$$

Let's compute each multiplication by hand:

1. Multiply the first two words:
   $$0.50 \times 0.40 = 0.20$$

2. Multiply by the third word:
   $$0.20 \times 0.30 = 0.06$$

3. Multiply by the fourth word:
   $$0.06 \times 0.90 = \mathbf{0.054}$$

The full sentence has an overall probability of <mark><strong>0.054</strong></mark> (or <mark><strong>5.4%</strong></mark>).

---

### The Paradox of 5.4%: Why Does It Look So Low, and Why Is It Actually Huge?

At first glance, a probability of <mark><strong>5.4%</strong></mark> looks tiny. You might wonder: *"If the model is so smart, shouldn't the probability of 'I love ice cream' be 80% or 90%?"*

Here is the deep mathematical secret: **In sequence probability, 5.4% is an astonishingly massive number.**

Let's see why:

#### 1. The 256-Sentence Universe (Combinatorial Explosion)
In our tiny toy language with only 4 words ($|V| = 4$), how many unique 4-word sentences can you write?

$$\text{Total Possible Sentences} = |V|^T = 4^4 = 256 \text{ sentences}$$

These include sentences like:
- <samp>"I I I I"</samp>
- <samp>"cream cream cream cream"</samp>
- <samp>"love ice cream I"</samp>
- <samp>"I love ice cream"</samp>
- ...and 252 other candidate combinations!

#### 2. Comparison With Pure Random Chance
If a child randomly picked 4 word blocks out of the box, every sentence would have an identical chance of:

$$P_{\text{random}} = \frac{1}{|V|^T} = \frac{1}{256} \approx 0.003906 \quad (0.39\%)$$

Our language model assigns <mark><strong>5.4%</strong></mark> to <samp>"I love ice cream"</samp>. Let's compare our model to random chance:

$$\frac{5.4\%}{0.39\%} \approx \mathbf{13.8\times \text{ higher than random chance!}}$$

In a crowded stadium of 256 competing sentences, one single sentence captured over 5% of all the probability mass in the entire universe. That is an overwhelming statistical endorsement.

#### 3. What Happens in Real LLMs? The Microscopic Decay & Log-Space
Now imagine a production LLM like GPT-4 or LLaMA-3:
- The vocabulary contains about $|V| \approx 100{,}000$ tokens.
- A typical paragraph has $T = 50$ tokens.

The number of possible 50-token sequences is:

$$|V|^T = 100{,}000^{50} = (10^5)^{50} = 10^{250}$$

By comparison, the entire observable universe contains only about $10^{80}$ atoms!

When you multiply 50 numbers that are each less than 1 (for instance, even if the model were 90% confident at every step, $0.90^{50} \approx 0.00515$; if it is 30% confident, $0.30^{50} \approx 10^{-26}$), the raw joint probability becomes microscopic.

**The Hardware Engineering Problem:**
If a computer chip tries to store a number like $10^{-250}$ using standard floating-point hardware (`float32`), the number is too tiny to represent. The chip encounters **arithmetic underflow** and rounds the number down to absolute $0.0$!

**The Mathematical Solution (Log-Probabilities):**
To prevent this underflow, AI researchers never multiply probabilities directly. Instead, they convert probabilities into logarithms. Because $\log(a \times b) = \log(a) + \log(b)$, dangerous multiplication turns into safe, stable addition:

$$\log P(w_1, w_2, \dots, w_T) = \sum_{t=1}^T \log P(w_t \mid w_{<t})$$

We will explore this logarithmic superpower in depth when we train models with **Cross-Entropy Loss** in Module 6!

---

<h2 id="step-6">Step 6: Core Takeaway</h2>

> [!TIP] Core Takeaway
> A Large Language Model has no magical human consciousness. At its core, it is a **probabilistic guessing machine**:
> 
> At every single position, it computes a probability distribution over its vocabulary $V$, selects a candidate word, appends that word to its prompt, and repeats.
> 
> But this brings us to the next big mystery: **Computers cannot understand raw letters or English words. How does an LLM convert words into numbers that silicon chips can compute?**

---

<nav aria-label="Chapter Navigation">
  <p>
    <a href="../index.html">Home / Curriculum Overview</a> &nbsp;|&nbsp; 
    <strong>Next Chapter:</strong> <a href="../01-vectors-and-spaces/index.html">Chapter 01: The Word Map (Vectors &amp; Embeddings) &rarr;</a>
  </p>
</nav>
