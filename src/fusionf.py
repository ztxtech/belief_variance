from dstz.core.distribution import Evidence
from dstz.evpiece.dual import ds_rule
from dstz.evpiece.single import shafer_discounting, pignistic_probability_transformation

def decision(evidence):
    """
    Determine the key with the maximum value from an Evidence object.

    Args:
        evidence (Evidence): An Evidence object containing key-value pairs.

    Returns:
        The value of the key with the maximum value in the Evidence object.
    """
    max_value = float('-inf')
    max_key = None
    for key, value in evidence.items():
        if value > max_value:
            max_value = value
            max_key = key
    return max_key.value

def murphy(mass_distributions, weights):
    """
    Apply the Murphy's method for evidence combination.

    Args:
        mass_distributions (list): A list of Evidence objects representing mass distributions.
        weights (list): A list of weights corresponding to each mass distribution.

    Returns:
        The result of the decision process after evidence combination.
    """
    weighted_mass = {}
    for index, mass in enumerate(mass_distributions):
        for key in mass.keys():
            if key in weighted_mass:
                weighted_mass[key] += mass[key] * weights[index]
            else:
                weighted_mass[key] = mass[key] * weights[index]
    weighted_mass = Evidence(weighted_mass)
    combined_result = weighted_mass
    for _ in range(len(mass_distributions) - 1):
        combined_result = ds_rule(combined_result, weighted_mass)
    pignistic_result = pignistic_probability_transformation(combined_result)
    decision_result = decision(pignistic_result)
    return decision_result
