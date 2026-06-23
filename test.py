# import requests

# # The URL of the file
# url = "https://huggingface.co/datasets/ankitdhiman/haryanvi-tts/resolve/main/metadata.csv"

# # The name you want the file to have locally
# filename = "metadata.csv"

# try:
#     # Send a GET request to the URL
#     response = requests.get(url)
    
#     # Check if the request was successful
#     response.raise_for_status()
    
#     # Write the content to a local file
#     with open(filename, "wb") as file:
#         file.write(response.content)
        
#     print(f"Successfully downloaded: {filename}")

# except requests.exceptions.RequestException as e:
#     print(f"An error occurred: {e}")