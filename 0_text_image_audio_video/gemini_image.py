import os
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from pathlib import Path
import pprint

os.environ.pop("SSL_CERT_FILE", None) 

def generate_image(prompt, output_path):
    """
    Generate an image using Gemini 2.0 Flash Preview Image Generation model and save it to output_path.

    Args:
        prompt (str): The image generation prompt.
        output_path (str or Path): Path to save the generated image (e.g., output/img/penguin/penguin_1.png).
    """
    try:
        # Convert output_path to Path object
        output_path = Path(output_path)

        # Validate and create output directory if needed
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"No write permission for directory: {output_dir}")
        print(f"📁 Output directory is ready: {output_dir}")
        print(f"💬 Sending prompt to Gemini: {prompt[:120]}...")

        # Initialize Gemini client
        api_key = os.getenv("GOOGLE_IMAGE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_IMAGE_API_KEY environment variable not set.")
        client = genai.Client(api_key=api_key)

        print("🔑 API client initialized.")

        # Request image generation
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"]
            )
        )

        found_image = False

        # Parse and save image
        for part in response.candidates[0].content.parts:
            if part.text:
                print(f"ℹ️ Text response: {part.text.strip()}")

            elif part.inline_data and part.inline_data.mime_type.startswith("image/"):
                print(f"🖼️ Found image part with MIME type: {part.inline_data.mime_type}")
                try:
                    image = Image.open(BytesIO(part.inline_data.data))
                    image.save(output_path)
                    print(f"✅ Image saved successfully at: {output_path}")
                    found_image = True
                    break
                except Exception as e:
                    print(f"❌ Error saving image: {str(e)}")

        if not found_image:
            print("❌ No image found in response. Dumping full response:")
            pprint.pprint(response.to_dict())

    except PermissionError as e:
        print(f"❌ Permission error: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

# from google import genai
# from google.genai import types
# from PIL import Image
# from io import BytesIO
# from pathlib import Path
# import os



# def test_gemini_image(prompt, output_path):
#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     api_key = os.getenv("GOOGLE_IMAGE_API_KEY")
#     if not api_key:
#         raise Exception("GOOGLE_IMAGE_API_KEY not set")

#     client = genai.Client(api_key=api_key)

#     print("🧠 Calling Gemini...")
#     response = client.models.generate_content(
#         model="gemini-2.0-flash-preview-image-generation",
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             response_modalities=["TEXT", "IMAGE"]
#         )
#     )
#     print("✅ Response received")

#     for part in response.candidates[0].content.parts:
#         if part.inline_data and part.inline_data.mime_type.startswith("image/"):
#             print("💾 Saving image...")
#             image = Image.open(BytesIO(part.inline_data.data))
#             image.save(output_path.with_suffix(".png"))
#             print("✅ Image saved")
#             return

#     print("❌ No image returned")

# # Run this
# test_gemini_image(
#     prompt="A cartoon penguin in a snowy landscape with bright colors and a big smile.",
#     output_path="output/test_penguin.png"
# )
