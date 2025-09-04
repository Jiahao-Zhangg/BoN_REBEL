import argparse
import numpy as np
from datasets import load_dataset
from scipy.spatial.distance import jensenshannon

def jensen_shannon_divergence(p, q):
    """
    Calculate Jensen-Shannon divergence between two probability distributions.
    """
    # Ensure inputs are numpy arrays
    p = np.array(p)
    q = np.array(q)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    
    # Normalize to ensure they sum to 1 (probability distributions)
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Calculate JS divergence using scipy
    return jensenshannon(p, q) ** 2

def SNTD(AB, BC, AC, type):
    """
    Calculate SNTD (Stochastic Non-Transitivity Divergence).
    
    Steps:
    1. Transform preferences to logit space: s = ln(x/(1-x))
    2. Reconstruct preferences using transitivity: AB_ = 1/(1+e^{-(sAC-sBC)})
    3. Calculate Jensen-Shannon divergences between original and reconstructed
    4. Return average JSD
    """
    # Transform to preference values first
    AB_pref = np.array(list(map(lambda x: transform_digit_to_preference(type, x), AB)))
    BC_pref = np.array(list(map(lambda x: transform_digit_to_preference(type, x), BC)))
    AC_pref = np.array(list(map(lambda x: transform_digit_to_preference(type, x), AC)))
    
    # Add small epsilon to avoid log(0) and log(inf)
    eps = 1e-10
    AB_pref = np.clip(AB_pref, eps, 1 - eps)
    BC_pref = np.clip(BC_pref, eps, 1 - eps)
    AC_pref = np.clip(AC_pref, eps, 1 - eps)
    
    # Step 1: Calculate logit transformations
    sAB = np.log(AB_pref / (1 - AB_pref))
    sBC = np.log(BC_pref / (1 - BC_pref))
    sAC = np.log(AC_pref / (1 - AC_pref))
    
    # Step 2: Reconstruct preferences using transitivity constraints
    # AB_ = 1/(1+e^{-(sAC-sBC)})
    AB_ = 1 / (1 + np.exp(-(sAC - sBC)))
    # BC_ = 1/(1+e^{-(sAB-sAC)})  
    BC_ = 1 / (1 + np.exp(-(sAB - sAC)))
    # AC_ = 1/(1+e^{-(sAB+sBC)})
    AC_ = 1 / (1 + np.exp(-(sAB + sBC)))
    
    # Step 3: Calculate Jensen-Shannon divergences
    jsd_AB = jensen_shannon_divergence(AB_pref, AB_)
    jsd_BC = jensen_shannon_divergence(BC_pref, BC_)
    jsd_AC = jensen_shannon_divergence(AC_pref, AC_)
    
    # Step 4: Return average JSD
    avg_jsd = (jsd_AB + jsd_BC + jsd_AC) / 3
    
    return avg_jsd, jsd_AB, jsd_BC, jsd_AC

def PNT(AB, BC, AC, type):
    MAP_intransitivity = {(0,0,0):0, (0,0,1):1,(0,1,0):0, (0,1,1):0,(1,0,0):0, (1,0,1):0, (1,1,0):1, (1,1,1):0}
    AB = list(map(lambda x: transform_digit_to_preference(type, x), AB))
    BC = list(map(lambda x: transform_digit_to_preference(type, x), BC))
    AC = list(map(lambda x: transform_digit_to_preference(type, x), AC))
    intransitivity = []
    for i in range(len(AB)):
        print(AB[i], BC[i], AC[i])
        if AB[i] == 0.5:
            if BC[i]==AC[i]:
                intransitivity.append(0)
            else:
                intransitivity.append(1)
        elif BC[i] == 0.5:
            if AB[i]==AC[i]:
                intransitivity.append(0)
            else:
                intransitivity.append(1)
        elif AC[i] == 0.5:
            if AB[i]==1-BC[i]:
                intransitivity.append(0)
            else:
                intransitivity.append(1)
        else:
            intransitivity.append(MAP_intransitivity[(AB[i],BC[i],AC[i])])
    return intransitivity

    
def process_dataset(dataset):
    """
    Process dataset to extract and flatten preference columns.
    Returns three lists: AB, BC, AC
    """
    AB = []
    BC = []
    AC = []
    
    for item in dataset:
        # Flatten lists from each column
        AB.extend(item['response_0_1_judged_preference_majority'])
        BC.extend(item['response_0_2_judged_preference_majority']) 
        AC.extend(item['response_1_2_judged_preference_majority'])
    
    return AB, BC, AC

def transform_digit_to_preference(type, digit_scores):
    if type == "5score":
        return digit_scores/4
    elif type == "ternary":
        MAP = {'A': 1, 'B': 0, 'Tie': 0.5}
        return MAP[digit_scores]

def main():
    parser = argparse.ArgumentParser(description='Check intransitivity in preference data')
    parser.add_argument('--input_repo', type=str, required=True, 
                       help='HuggingFace repository name to load dataset from')
    parser.add_argument('--type', type=str, required=True, choices=['5score', 'ternary'],
                       help='Type of preference data (5score or ternary)')
    
    args = parser.parse_args()
    
    print(f"Loading dataset from: {args.input_repo}")
    dataset = load_dataset(args.input_repo)
    
    # Use the train split by default, adjust if needed
    if 'train' in dataset:
        dataset = dataset['train']
    else:
        # Use the first available split
        dataset = dataset[list(dataset.keys())[0]]
    
    print("Processing dataset...")
    AB, BC, AC = process_dataset(dataset)
    
    print(f"Extracted {len(AB)} AB preferences, {len(BC)} BC preferences, {len(AC)} AC preferences")
    
    print("Calculating intransitivity...")
    intransitivity_vector = PNT(AB, BC, AC, args.type)
    
    # Calculate ratio of 1s (True values) in the intransitivity vector
    intransitivity_ratio = np.mean(intransitivity_vector)
    
    print(f"Intransitivity ratio: {intransitivity_ratio:.4f}")
    print(f"Total intransitive cases: {np.sum(intransitivity_vector)} out of {len(intransitivity_vector)}")
    
    # print("\nCalculating SNTD...")
    # avg_jsd, jsd_AB, jsd_BC, jsd_AC = SNTD(AB, BC, AC, args.type)
    
    # print(f"SNTD (average Jensen-Shannon divergence): {avg_jsd:.4f}")
    # print(f"JSD(AB, AB_): {jsd_AB:.4f}")
    # print(f"JSD(BC, BC_): {jsd_BC:.4f}")
    # print(f"JSD(AC, AC_): {jsd_AC:.4f}")

if __name__ == "__main__":
    main()

