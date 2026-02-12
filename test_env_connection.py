"""
Test script to verify .env file connection
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if HF token is loaded
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    # Mask the token for security
    masked_token = hf_token[:10] + "..." + hf_token[-5:] if len(hf_token) > 15 else "***"
    print("✅ SUCCESS: HuggingFace token loaded from .env file")
    print(f"   Token (masked): {masked_token}")
    print(f"   Token length: {len(hf_token)} characters")
else:
    print("❌ ERROR: No HF_TOKEN found in .env file")
    print("   Please check your .env file contains: HF_TOKEN=your_token_here")
