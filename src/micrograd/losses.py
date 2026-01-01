from micrograd import Value 

def softmax(logits):
    """
    logits: list[Value] (scores, not probabilities)
    returns: list[Value] probs, sum ~ 1
    """
    if not logits:
        raise ValueError("softmax() requires at least one logit")
    
    # TODO 1: compute a float "shift" for numerical stability
    # HINT: use .data because we only want a constant shift
    shift = float(max(l.data for l in logits))
    
    # TODO 2: compute exponentials of shifted logits
    # exps[i] should be Value = exp(logits[i] - shift)
    exps = [(logits[i] - shift).exp() for i in range(len(logits))]  # list[Value]

    # TODO 3: compute the denominator = sum of exponentials (as a Value)
    denom = sum(exps[1:], exps[0]) # 
    # denom = sum(exps): Should be fine but code above avoids the 0 + Value()

    # TODO 4: compute probabilities by dividing each exp by denom
    probs = [e / denom for e in exps]  # list[Value]

    return probs

def nll_loss(probs, target_index):
    # TODO 1: pick the probability assigned to the correct class
    p = probs[target_index]

    return -(p.log())