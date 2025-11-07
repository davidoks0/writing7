"""
Inference script for book-text matching classifier.

Given a book ID and a text snippet, predict whether the snippet belongs to that book.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BookMatcher:
    """Class for making predictions on book-text pairs."""
    
    def __init__(self, model_path: str):
        """Load model and tokenizer."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded model from {model_path} on {self.device}")
    
    def predict(self, text1: str, text2: str) -> dict:
        """
        Predict whether two text chunks come from the same book.
        
        Args:
            text1: First text chunk
            text2: Second text chunk
        
        Returns:
            dict with 'same_book' (bool) and 'confidence' (float)
        """
        # Tokenize
        inputs = self.tokenizer(
            text1,
            text2,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        # Get prediction
        same_book_prob = probs[0][1].item()
        same_book = same_book_prob > 0.5
        confidence = same_book_prob if same_book else (1 - same_book_prob)
        
        return {
            'same_book': same_book,
            'confidence': confidence,
            'probability': same_book_prob
        }
    
    def predict_batch(self, pairs: list) -> list:
        """Predict on multiple pairs."""
        results = []
        for text1, text2 in pairs:
            result = self.predict(text1, text2)
            results.append(result)
        return results


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict whether text belongs to a book')
    parser.add_argument('--model', type=str, default='models/book_matcher/final',
                       help='Path to trained model')
    parser.add_argument('--text1', type=str, required=True,
                       help='First text chunk')
    parser.add_argument('--text2', type=str, required=True,
                       help='Second text chunk')
    
    args = parser.parse_args()
    
    # Load model
    matcher = BookMatcher(args.model)
    
    # Make prediction
    result = matcher.predict(args.text1, args.text2)
    
    print(f"\nPrediction: {'SAME BOOK' if result['same_book'] else 'DIFFERENT BOOKS'}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probability: {result['probability']:.4f}")
    
    print(f"\nText 1: {args.text1[:200]}...")
    print(f"Text 2: {args.text2[:200]}...")


if __name__ == '__main__':
    main()

