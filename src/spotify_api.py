from .config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET

def has_credentials() -> bool:
    return bool(SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET)

def get_client():
    """Return a Spotipy client if creds exist, else None."""
    if not has_credentials():
        return None
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    auth = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
    return spotipy.Spotify(auth_manager=auth)
