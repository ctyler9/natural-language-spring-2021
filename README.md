# natural-language-spring-2021
WSB analysis



I think this is a reasonable project but may be a bit on the ambitious / risky side since it sounds like you will need to construct new datasets (both from WSB and stock prices). I think this is doable, but still there is a question of how you would link the posts on WSB to specific stocks. I guess the ticker symbols would probably make this relatively easy...

Another issue is the difference in domain - it's not clear that a model trained on movie sentiment will work well for financial sentiment. I believe there are some datasets and tools for measuring sentiment in finance which may be more appropriate (e.g. http://kaichen.work/?p=399).

Anyway, I think this sounds like a feasible project, but might need a bit of careful thought to pull it off in a meaningful way.

Another option would be to focus more on the methods side for financial sentiment. For example, it looks like there is at least one SemEval dataset that might be available for this, but you need to fill out a form to request access:
https://alt.qcri.org/semeval2017/task5/index.php?id=data-and-tools

Also, BTW, just to point out, one of my (now graduated) Ph.D. students has some work on analyzing text associated with financial earnings forecasts, in case this helps to give some inspiration:
https://www.aclweb.org/anthology/2020.acl-main.473.pdf

I hope this helps!

-Alan
