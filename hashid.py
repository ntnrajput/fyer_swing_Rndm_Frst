import hashlib
client_id = "VE3CCLJZWA-100"  # Replace with your actual Client ID
hashed = hashlib.sha256(client_id.encode()).hexdigest()
print("FYERS_APP_ID_HASH =", hashed)
