import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image, ImageEnhance
import io
import os
import logging
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # Adjust this to your frontend domain in production
    allow_origins=["https://polite-rock-0f7ea8200.5.azurestaticapps.net/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set your Azure Computer Vision credentials
subscription_key = os.getenv('sub_key')
endpoint = os.getenv('end_point')
groq_api_key = os.getenv('groq_api_key')

# Initialize GROQ client
client = Groq(api_key=groq_api_key)


if not subscription_key or not endpoint or not groq_api_key:
    raise ValueError(
        "Environment variables for Azure and Groq API keys must be set")
# Initialize Computer Vision client
computervision_client = ComputerVisionClient(
    endpoint, CognitiveServicesCredentials(subscription_key))

# Function to enhance and analyze image


def enhance_and_analyze_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    original_width, original_height = image.size
    # Convert image to RGB if it's not already in that mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    enhancer = ImageEnhance.Contrast(image)
    image_enhanced = enhancer.enhance(2.0)
    image_stream = io.BytesIO()
    image_enhanced.save(image_stream, format='JPEG')
    image_stream.seek(0)
    read_response = computervision_client.read_in_stream(
        image_stream, raw=True)
    read_operation_location = read_response.headers["Operation-Location"]
    operation_id = read_operation_location.split("/")[-1]

    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
            break

    text_results = []
    if read_result.status == OperationStatusCodes.succeeded:
        for page in read_result.analyze_result.read_results:
            for line in page.lines:
                min_x = min(line.bounding_box[0::2])
                min_y = min(line.bounding_box[1::2])
                max_x = max(line.bounding_box[0::2])
                max_y = max(line.bounding_box[1::2])

                bounding_box = {
                    "left": (min_x / original_width) * 100,
                    "top": (min_y / original_height) * 100,
                    "width": ((max_x - min_x) / original_width) * 100,
                    "height": ((max_y - min_y) / original_height) * 100
                }
                text_results.append({
                    "text": line.text,
                    "bounding_box": bounding_box
                })

    return text_results

# Function to classify text using Groq API


def classify_text_groq(text_to_classify):
    try:
        # Format the prompt for GROQ
        prompt = f"""
        Extract and classify the following text into the required details:

        Extract the following details:
        - Known Context: What is the context or purpose of the system described?
        - Brand/Mark: If there is any specific brand or mark mentioned, extract it.
        - Keywords: Extract key terms that describe the components and their roles.
        - Detected Objects: List the classes and their associated attributes and methods.

        Text to Analyze:
        {text_to_classify}

        Provide the output in the following format:

        Known Context = [context]

        Brand/Mark = [brand]

        Keywords = [keywords]

        Detected Objects = [object structure with attributes and methods]
        """

        # Call GROQ API to classify the text
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
        )

        # Extract the result
        result = chat_completion.choices[0].message.content.strip()
        return result
    except Exception as e:
        logging.error(f"Error classifying text with GROQ: {e}")
        raise

        # for result in text_results:
        #     bbox = result["bounding_box"]
        #     top_left = (int(bbox["left"]), int(bbox["top"]))
        #     bottom_right = (int(bbox["left"] + bbox["width"]),
        #                     int(bbox["top"] + bbox["height"]))
        #     cv2.rectangle(image_cv2, top_left, bottom_right, (0, 255, 0), 2)


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image_bytes = io.BytesIO(image_data).getvalue()

        # Enhance the image and extract text with bounding boxes
        text_results = enhance_and_analyze_image(image_data)

        # Extract text to be classified
        detected_text = " ".join([result["text"] for result in text_results])

        # Classify the extracted text using Groq
        classification_results = classify_text_groq(detected_text)

        # Extract bounding boxes
        bounding_boxes = [result["bounding_box"] for result in text_results]

        return {
            "bounding_boxes": bounding_boxes,  # Return the bounding boxes
            "classified_text": classification_results,  # Return the classified text
        }
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Default to 8000 if PORT is not set
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("your_app_module:mainapp", host="0.0.0.0", port=port)
