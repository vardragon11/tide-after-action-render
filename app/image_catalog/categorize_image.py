import spacy

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

def categorize_prompt(prompt: str):
    doc = nlp(prompt)

    categories = {
        "objects": [],
        "people": [],
        "places": [],
        "adjectives": [],
        "verbs": [],
        "named_entities": []
    }

    for token in doc:
        if token.pos_ == "NOUN":
            categories["objects"].append(token.text)
        elif token.pos_ == "PROPN":
            categories["people"].append(token.text)
        elif token.pos_ == "ADJ":
            categories["adjectives"].append(token.text)
        elif token.pos_ == "VERB":
            categories["verbs"].append(token.text)

    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            categories["places"].append(ent.text)
        categories["named_entities"].append((ent.text, ent.label_))

    # Remove duplicates
    for key in categories:
        categories[key] = list(set(categories[key]))

    return categories