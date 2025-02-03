import re
from transformers import CLIPTokenizer
from Levenshtein import distance as levenshtein_distance

# Load CLIP tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def preprocess_text(text):
    """Clean and normalize text by removing special characters and extra spaces."""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()).strip()

def get_token_count(text):
    """Returns the number of tokens in the text using CLIP tokenizer."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def generate_ngrams(text, min_n, max_n):
    """Generate n-grams from min_n to max_n based on tokenized word count."""
    cleaned_text = preprocess_text(text)
    words = cleaned_text.split()

    ngram_candidates = []
    for n in range(min_n, max_n + 1):
        if n > len(words):
            break  # Avoid generating n-grams larger than available words
        ngram_candidates.extend([' '.join(words[i:i + n]) for i in range(len(words) - n + 1)])
    
    return ngram_candidates

def levenshtein_match(generated_text, ground_truth):
    """Find the substring with the closest Levenshtein distance for varying n-gram sizes."""
    ground_truth_tokens = get_token_count(ground_truth)
    ngram_candidates = generate_ngrams(generated_text, min_n=1, max_n=ground_truth_tokens + 1)

    # Rank candidates by Levenshtein distance
    ranked_matches = sorted(ngram_candidates, key=lambda x: levenshtein_distance(x, ground_truth))
    print(ranked_matches)
    best_match = ranked_matches[0]
    best_score = levenshtein_distance(best_match, ground_truth)
    return best_match, best_score

def ngram_levenshtein(generated_list, ground_truth_list):
    """Process multiple generated strings and compute average Levenshtein distance."""
    assert len(generated_list) == len(ground_truth_list), "Lists must have the same length."

    total_distance = 0
    results = []

    for generated, truth in zip(generated_list, ground_truth_list):
        # handling generated empty strings 
        if generated == '':
            best_match, distance = '', len(truth)
        else : 
            best_match, distance = levenshtein_match(generated, truth)
        total_distance += distance
        results.append((truth, best_match, distance))

    average_distance = total_distance / len(generated_list)
    return results, average_distance

# # Example usage
# generated_texts = [
#     "jey sniftlag a 0g8 [ract vaxe gu' g ysi ei# wrz",
#     "thi iz a sampl text wth erors",
#     "cllp tokenzr is usd hre"
# ]

# ground_truths = [
#     "sniff bag",
#     "this is a sample text with errors",
#     "clip tokenizer is used here"
# ]

# matches, avg_distance = process_text_list(generated_texts, ground_truths)

# print("\n=== Closest Matches and Scores ===")
# for truth, match, dist in matches:
#     print(f"Ground truth: {truth} | Best match: {match} | Levenshtein distance: {dist}")

# print(f"\nAverage Levenshtein Distance: {avg_distance:.2f}")

# generated_text = [
#     "twezers", "road", "fielld", "", "lappp", "t", "quddudr % ye 2", "rolleers balde", "travel", 
#     "story", "garage", "boom", "viley", "moniitor", "clipper", "cub uht", "mouse kiwet", 
#     "ieeduu newvappe", "2 allley", "octpus sutraiis", "tie", "", "pagraagh", "", "shampooo", 
#     "wh is. segie", "z<oiy alot-feren huraane nutt, more ferelanteil", 
#     "monkey inse nskc $ iheuy fus you we ins tiend wid iux rddr &ofc ko:", "e", "@jindew oinger", 
#     "", "marker stcntt", "museum", "wiale", "crow", "celeebratiion", "robe cwart", "bju", 
#     "garden uuth ~bnudlie", "riake", "trail", "ioe acioune (o onler", "+", "island", "schedule", 
#     "paintish we |6 veke", "bear", "aul bese", "roa d", "tune mirror", "", "dustaan d xone", 
#     "yotu ot anc me {0 jout 4 yoid oui ntyou .", "balnoo1", "pencii penci", "lbary r @ toe mie 'y vry i1 mou rro", 
#     "ak teldly bear", "", "truck", "toy", "village", "", "rake", "saw", "weecnch", "r", "", 
#     "letnaine", "065", "", "holiday", "me ad thqid's", "nail", "", "sore € tre bell catd", 
#     "12 0 2 5 3 81 6", "", "animal crovise", "snane", "wharve dorrcere wedding", "", "iled stualcek", 
#     "pieg", "rcksse", "", "buttefly", "curtain ccuch ck", "", "ocean", "", 
#     "geckere is its jstikkd:", "bare", "barrn", "(ucr v", "bajties", "yard youn nor los kewen", 
#     "shortes", "razor", "sand", "sea", "hilll"
# ]

# if __name__=="__main__":
#     generated_text = ['', 'twezers', 'road', 'fielld', '', 'lappp', 't', 'quddudr % ye 2', 'rolleers balde', 'travel', 'story', 'garage', 'boom', 'viley', 'moniitor', 'clipper', 'cub uht', 'mouse kiwet', 'ieeduu newvappe', '2 allley', 'octpus sutraiis', 'tie', '', 'pagraagh', '', 'shampooo', 'wh is. segie', 'z<oiy alot-feren huraane nutt, more ferelanteil', 'monkey inse nskc $ iheuy fus you we ins tiend wid iux rddr &ofc ko:', 'e', '@jindew oinger', '', 'marker stcntt', 'museum', 'wiale', 'crow', 'celeebratiion', 'robe cwart', 'bju', 'garden uuth ~bnudlie', 'riake', 'trail', 'ioe acioune (o onler', '', 'island', 'schedule', 'paintish we |6 veke', 'bear', 'aul bese', 'roa d', 'tune mirror', '', 'dustaan d xone', 'yotu ot anc me {0 jout 4 yoid oui ntyou .', 'balnoo1', 'pencii penci', "lbary r @ toe mie 'y vry i1 mou rro", 'ak teldly bear', 'truck', 'toy', 'village', '', 'rake', 'saw', 'weecnch', 'r', '', 'letnaine', '065', 'holiday', "me ad thqid's", 'nail', '', 'sore € tre bell catd', '12 0 2 5 3 81 6', '', 'animal crovise', 'snane', 'wharve dorrcere wedding', '', 'iled stualcek', 'pieg', 'rcksse', '', 'buttefly', 'curtain ccuch ck', '', 'ocean', '', 'geckere is its jstikkd:', 'bare', 'barrn', '(ucr v', 'bajties', 'yard youn nor los kewen', 'shortes', 'razor', 'sand', 'sea', 'hilll']

#     words = ['Sun', 'Tweezers', 'Road', 'Field', 'Insect', 'Laptop', 'Radio', 'Calendar', 'Rollerblade', 'Travel', 'Story', 'Garage', 'Broom', 'Valley', 'Monitor', 'Clipper', 'Cub', 'Mouse', 'Newspaper', 'Alley', 'Octopus', 'Tie', 'Deer', 'Paragraph', 'Duck', 'Shampoo', 'Seagull', 'Wheelbarrow', 'Monkey', 'Word', 'Window', 'Glass', 'Marker', 'Museum', 'Whale', 'Crow', 'Celebration', 'Robe', 'Robot', 'Garden', 'Lizard', 'Trail', 'Necklace', 'Hospital', 'Island', 'Schedule', 'Paintbrush', 'Bear', 'Easel', 'Road', 'Mirror', 'Box', 'Dustpan', 'Poem', 'Balloon', 'Pencil', 'Library', 'Teddy Bear', 'Truck', 'Toy', 'Village', 'Zebra', 'Rake', 'Saw', 'Wrench', 'Puzzle', 'Ship', 'Valve', 'Pen', 'Holiday', 'Parachute', 'Nail', 'Root', 'Belt', 'Clock', 'Tunnel', 'Animal', 'Shark', 'Wedding', 'Helicopter', 'Starfish', 'Pigeon', 'Pickaxe', 'Seahorse', 'Butterfly', 'Curtain', 'Flamingo', 'Ocean', 'Parrot', 'Jacket', 'Bluebird', 'Barn', 'Camera', 'Wagon', 'Yard', 'Shorts', 'Razor', 'Sand', 'Sea', 'Hill']
#     words = [word.lower() for word in words]
    
#     results, average_distance = ngram_levenshtein(generated_text, words)
#     print(average_distance)