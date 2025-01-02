
def predict_headline_class(headline):
    """
    Predicts the category of a Bangla news headline.

    Args:
        headline (str): The Bangla news headline.

    Returns:
        str: The predicted category.
    """
    # Tokenize the input headline
    inputs = tokenizer(
        headline,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted label
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return LABELS[predicted_class]

if __name__ == "__main__":
    # Example usage
    headline = input("Enter a Bangla news headline: ")
    category = predict_headline_class(headline)
    print(f"Predicted Category: {category}")
