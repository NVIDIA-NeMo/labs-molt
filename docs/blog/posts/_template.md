<!--
  Template for a Molt blog post. Copy this file to blog/posts/<slug>.md, write the post,
  then add an entry with that slug at the top of blog/posts.js.
  Preview this template at blog/?p=_template. This comment is not shown on the site.
-->

# Post title

Open with the point of the post in two or three sentences. The title, date, tag, and
summary shown above the post come from its entry in `blog/posts.js`.

## A section heading

Body text supports **bold**, *italic*, `inline code`, and
[links](https://github.com/NVIDIA-NeMo/labs-molt).

- A list item
- Another list item

> A note or a pull quote.

## Code

```python
from molt.agents import Env, Result, StepEnvRunner


class MathEnv(Env):
    async def step(self, state) -> Result:
        reward = grade(state["action_text"], state["label"])
        return Result(reward=reward, terminated=True)
```

## Math

Inline math such as $A_i = R_i - \frac{1}{G}\sum_{j} R_j$ sits in the sentence. Display
math goes in a `math` block, the same way GitHub renders it:

```math
\widehat{\mathcal L}(\theta) = -\frac{1}{B}\sum_{i=1}^{B}\frac{1}{T_i}\sum_{t=1}^{T_i}
A_i\,\frac{\pi_\theta(a_{it}\mid h_{it})}{\mu_i(a_{it}\mid h_{it})}
```

## Figures and tables

Put images in `blog/posts/images/` and reference them relative to this file, for example
`![Learning curve](images/learning-curve.png)`.

| Setting | Value |
|---|---|
| Model | Qwen3-4B |
| Estimator | `reinforce_baseline` |
