try:
    from google import genai
    print("SUCCESS: Google GenAI SDK is installed and importable.")
except ImportError as e:
    print(f"ERROR: Failed to import google.genai: {e}")
except Exception as e:
    print(f"ERROR: An unexpected error occurred: {e}")
