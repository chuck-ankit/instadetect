import os
import sys

# Add the app directory to Python path
app_dir = os.path.join(os.path.dirname(__file__), "app")
sys.path.append(app_dir)

# Import and run the Streamlit app
from main import main

if __name__ == "__main__":
    main()
