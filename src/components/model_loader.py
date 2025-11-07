import os
import zipfile
import pickle
import torch
import json
from transformers import BertModel, AutoTokenizer
import torch.nn as nn

from src.common.logger import logger
from src.common.exception import CustomException


class MultitaskBERT(nn.Module):
    def __init__(self, num_sentiment_labels, num_intent_labels, num_urgency_labels, num_topic_labels):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.sentiment_classifier = nn.Linear(hidden_size, num_sentiment_labels)
        self.intent_classifier = nn.Linear(hidden_size, num_intent_labels)
        self.urgency_classifier = nn.Linear(hidden_size, num_urgency_labels)
        self.topic_classifier = nn.Linear(hidden_size, num_topic_labels)

    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return {
            'sentiment': self.sentiment_classifier(pooled_output),
            'intent': self.intent_classifier(pooled_output),
            'urgency': self.urgency_classifier(pooled_output),
            'topic': self.topic_classifier(pooled_output)
        }
    

class ModelLoader:
    def __init__(self, zip_path: str = "model/customer_feedback_model.zip", extract_path: str = "model/extracted"):
        self.zip_path = zip_path
        self.extract_path = extract_path
        self.analyzer = None
        logger.info("ModelLoader initialized")


    def unzip_model(self):
        """Unzips the model archive to the specified extraction path."""
        try:
            if not os.path.exists(self.zip_path):
                raise CustomException(f"Model zip file not found at {self.zip_path}")
            
            # create extraction directory if it doesn't exist
            os.makedirs(self.extract_path, exist_ok=True)

            # Extract the zip file
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_path)
            
            logger.info(f"Model unzipped successfully to {self.extract_path}")
            return True

        except Exception as e:
            logger.error(f"Error unzipping model: {e}")
            raise CustomException(f"Failed to unzip model: {e}")
        

    def load_model(self):
        """Load the model and related components"""

        try: 
            if not os.path.exists(self.extract_path):
                self.unzip_model()
            
            # Load configuration
            config_path = os.path.join(self.extract_path, "model_config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Load label encoders
            encoders_path = os.path.join(self.extract_path, "label_encoders.pkl")
            with open(encoders_path, 'rb') as f:
                label_encoders = pickle.load(f)
            
            # Load tokenizer
            tokenizer_path = os.path.join(self.extract_path, "tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            # Initialize model
            model = MultitaskBERT(
                num_sentiment_labels=config['num_sentiment_labels'],
                num_intent_labels=config['num_intent_labels'],
                num_urgency_labels=config['num_urgency_labels'],
                num_topic_labels=config['num_topic_labels']
            )

            # Load model weights
            weights_path = os.path.join(self.extract_path, "model_weights.pth")
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            model.eval()

            # Create label decoders
            label_decoders = {}
            for task, encoder in label_encoders.items():
                label_decoders[task] = {v: k for k, v in encoder.items()}

            self.analyzer = {
                'model': model,
                'tokenizer': tokenizer,
                'label_decoders': label_decoders
            }

            logger.info("Model and components loaded successfully")
            return self.analyzer

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise CustomException(f"Failed to load model: {e}")
        
    

    def predict(self, text):
        """Make prediction on input text"""
        try:
            if self.analyzer is None:
                self.load_model()
            
            model = self.analyzer['model']
            tokenizer = self.analyzer['tokenizer']
            label_decoders = self.analyzer['label_decoders']
            
            # Tokenize input
            inputs = tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt", 
                return_token_type_ids=False
            )
            
            # Make prediction
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
            
            # Process results
            result = {'text': text}
            for task, logits in outputs.items():
                probs = torch.softmax(logits, dim=1)
                confidence, pred_idx = torch.max(probs, 1)
                result[task] = {
                    'label': label_decoders[task][pred_idx.item()],
                    'confidence': round(confidence.item(), 3)
                }
            
            logger.info(f"Prediction made for text: {text[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise CustomException(f"Prediction failed: {str(e)}")
