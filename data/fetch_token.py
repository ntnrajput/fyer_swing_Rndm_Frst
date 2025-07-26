# data/fetch_token.py (modified to use fyers_apiv3.SessionModel)

import webbrowser
from fyers_apiv3 import fyersModel
from config import FYERS_CLIENT_ID, FYERS_SECRET_ID, FYERS_REDIRECT_URI
from utils.logger import get_logger

logger = get_logger(__name__)

def launch_browser_login():
    """
    Use Fyers v3 SessionModel to generate login URL and open browser.
    """
    session = fyersModel.SessionModel(
        client_id=FYERS_CLIENT_ID,
        secret_key=FYERS_SECRET_ID,
        redirect_uri=FYERS_REDIRECT_URI,
        response_type="code",
        state="sample_state"
    )

    response = session.generate_authcode()

    if "url" in response:
        auth_url = response["url"]
        logger.info("Opening login page in browser...")
        webbrowser.open(auth_url)
    else:
        logger.error(f"Failed to generate login URL: {response}")
