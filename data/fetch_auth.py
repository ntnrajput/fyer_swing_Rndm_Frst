# refactored_fyers_swing/data/fetch_auth.py

from fyers_apiv3 import fyersModel
from config import FYERS_CLIENT_ID, FYERS_SECRET_ID, FYERS_REDIRECT_URI, TOKEN_PATH
from utils.logger import get_logger

logger = get_logger(__name__)

def generate_access_token(auth_code: str) -> str:
    """
    Exchange auth_code for access_token using FYERS API v3.
    """
    try:
        session = fyersModel.SessionModel(
            client_id=FYERS_CLIENT_ID,
            secret_key=FYERS_SECRET_ID,
            redirect_uri=FYERS_REDIRECT_URI,
            response_type="code",
            grant_type="authorization_code"
        )

        session.set_token(auth_code)
        response = session.generate_token()

        if "access_token" in response:
            access_token = response["access_token"]
            TOKEN_PATH.write_text(access_token)
            logger.info("Access token generated and saved.")
            return access_token
        else:
            logger.error(f"❌ Access token not found in response: {response}")
            return ""

    except Exception as e:
        logger.exception(f"Failed to generate access token: {e}")
        return ""


def get_saved_access_token() -> str:
    """
    Return saved access token if available.
    """
    if TOKEN_PATH.exists():
        return TOKEN_PATH.read_text().strip()
    logger.warning("⚠️ Access token file not found.")
    return ""
