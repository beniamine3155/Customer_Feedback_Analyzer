from src.components.model_loader import ModelLoader
from src.common.logger import logger
from src.common.exception import CustomException

class ModelTester:
    def __init__(self):
        self.model_loader = ModelLoader()
        logger.info("ModelTester initialized")
    
    def test_model(self):
        """Test the model with sample inputs"""
        try:
            logger.info("Starting model testing...")
            
            # Sample test cases
            test_cases = [
                "Your app is constantly crashing when I try to process payments! This is urgent!",
                "I love the new dashboard feature! It's made my work so much easier.",
                "How do I export my data to Excel? I can't find the option.",
                "The billing system charged me twice this month. This is unacceptable!",
                "Can you add dark mode to the mobile app? That would be amazing!",
                "Your customer service was terrible! I waited for hours with no help.",
            ]
            
            results = []
            for i, text in enumerate(test_cases, 1):
                try:
                    result = self.model_loader.predict(text)
                    results.append(result)
                    
                    print(f"\n{'='*60}")
                    print(f"TEST CASE {i}:")
                    print(f"{'='*60}")
                    print(f"Input: {text}")
                    print(f"Sentiment: {result['sentiment']['label']} (confidence: {result['sentiment']['confidence']:.3f})")
                    print(f"Intent: {result['intent']['label']} (confidence: {result['intent']['confidence']:.3f})")
                    print(f"Urgency: {result['urgency']['label']} (confidence: {result['urgency']['confidence']:.3f})")
                    print(f"Topic: {result['topic']['label']} (confidence: {result['topic']['confidence']:.3f})")
                    
                except Exception as e:
                    logger.error(f"Failed to process test case {i}: {str(e)}")
                    print(f"Test case {i} failed: {str(e)}")
            
            logger.info(f"Model testing completed. {len(results)}/{len(test_cases)} tests passed")
            return results
            
        except Exception as e:
            logger.error(f"Model testing failed: {str(e)}")
            raise CustomException(f"Model testing failed: {str(e)}")

# For quick testing
# if __name__ == "__main__":
#     tester = ModelTester()
#     tester.test_model()