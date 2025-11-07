from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
import uvicorn

from src.common.logger import logger
from src.common.exception import CustomException
from src.components.model_loader import ModelLoader


app = FastAPI(
    title="Customer Feedback Analyzer",
    description="An application to analyze customer feedback for sentiment, intent, urgency, and topic.",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global model loader instance
model_loader = None

class FeedbackRequest(BaseModel):
    text: str

# Example texts for the UI
EXAMPLE_TEXTS = [
    {"label": "Urgent Technical Issue", "text": "Your app is constantly crashing when I try to process payments! This is urgent!"},
    {"label": "Positive Feedback", "text": "I love the new dashboard feature! It's made my work so much easier."},
    {"label": "General Question", "text": "How do I export my data to Excel? I can't find the option."},
    {"label": "Billing Complaint", "text": "The billing system charged me twice this month. This is unacceptable!"},
    {"label": "Feature Suggestion", "text": "Can you add dark mode to the mobile app? That would be amazing!"},
    {"label": "Service Complaint", "text": "Your customer service was terrible! I waited for hours with no help."},
]

def get_business_recommendations(result: dict) -> list:
    """Generate business recommendations based on analysis results"""
    sentiment = result['sentiment']['label']
    intent = result['intent']['label']
    urgency = result['urgency']['label']
    topic = result['topic']['label']
    
    recommendations = []
    
    if urgency == 'high' and sentiment == 'negative':
        recommendations.extend([
            "IMMEDIATE: Escalate to senior support team",
            "Offer immediate apology and compensation", 
            "Resolve within 2 hours maximum",
            "Personal follow-up call required"
        ])
    elif intent == 'complaint':
        recommendations.extend([
            "PRIORITY: Route to complaints department",
            "Investigate issue thoroughly",
            "Document all interactions", 
            "Follow up within 8 hours"
        ])
    elif intent == 'suggestion' and sentiment == 'positive':
        recommendations.extend([
            "FEEDBACK: Thank customer for suggestion",
            "Forward to product development team",
            "Log in feature request database",
            "Consider for future updates"
        ])
    elif intent == 'question':
        recommendations.extend([
            "SUPPORT: Provide helpful documentation",
            "Offer step-by-step guidance",
            "Share relevant tutorials",
            "Respond within 24 hours"
        ])
    elif intent == 'appreciation':
        recommendations.extend([
            "POSITIVE: Thank customer sincerely", 
            "Consider asking for testimonial",
            "Share with team for morale boost",
            "Consider loyalty reward"
        ])
    else:
        recommendations.extend([
            "STANDARD: Process through support workflow",
            "Respond with empathy and understanding",
            "Acknowledge within 24 hours",
            "Monitor for escalation needs"
        ])
    
    # Add topic-specific recommendations
    if topic == 'technical':
        recommendations.append("Assign to technical support specialist")
    elif topic == 'billing':
        recommendations.append("Route to billing department specialist")
    elif topic == 'account':
        recommendations.append("Escalate to account management team")

    return recommendations


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model_loader
    try:
        logger.info("Starting up Customer Feedback Analyzer...")
        model_loader = ModelLoader()
        model_loader.load_model()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize during startup: {str(e)}")
        raise CustomException(f"Startup initialization failed: {str(e)}")
    

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main page with the feedback form"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "examples": EXAMPLE_TEXTS,
            "analyzing": False
        }
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_feedback(
    request: Request,
    feedback_text: str = Form(...)
):
    """Analyze customer feedback and return results"""
    try:
        logger.info(f"Received analysis request for text: {feedback_text[:50]}...")
        
        if model_loader is None:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "examples": EXAMPLE_TEXTS,
                    "error": "Model is not loaded. Please try again later.",
                    "feedback_text": feedback_text
                }
            )
        
        # Make prediction
        result = model_loader.predict(feedback_text)
        
        # Generate business recommendations
        recommendations = get_business_recommendations(result)
        
        logger.info(f"Analysis completed successfully for: {feedback_text[:50]}...")
        
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "result": result,
                "recommendations": recommendations
            }
        )
        
    except CustomException as e:
        logger.error(f"Custom error during analysis: {str(e)}")
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "examples": EXAMPLE_TEXTS,
                "error": f"Analysis error: {str(e)}",
                "feedback_text": feedback_text
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}")
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "examples": EXAMPLE_TEXTS,
                "error": "An unexpected error occurred. Please try again.",
                "feedback_text": feedback_text
            }
        )
    

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Customer Feedback Analyzer is running",
        "model_loaded": model_loader is not None
    }


@app.get("/test")
async def test_model():
    """Test endpoint to verify model is working"""
    try:
        test_text = "I love this product! It's absolutely amazing."
        result = model_loader.predict(test_text)
        
        return {
            "status": "success",
            "message": "Model is working correctly",
            "test_result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model test failed: {str(e)}")
    

@app.post("/api/analyze")
async def analyze_feedback_api(request: FeedbackRequest):
    """API endpoint for programmatic access"""
    try:
        if model_loader is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        result = model_loader.predict(request.text)
        
        return {
            "text": result['text'],
            "sentiment": result['sentiment'],
            "intent": result['intent'],
            "urgency": result['urgency'],
            "topic": result['topic']
        }
        
    except CustomException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
    

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )